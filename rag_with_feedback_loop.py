import os, sys
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

## --------------------------
## Data Models
## --------------------------

@dataclass
class Feedback:
    query: str
    response: str
    relevance: int = field(default=1, metadata={"description": "Score from 1-5"})
    quality: int = field(default=1, metadata={"description": "Score from 1-5"})
    comments: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "relevance": self.relevance,
            "quality": self.quality,
            "comments": self.comments
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        return cls(**data)


class ResponseModel(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")


## --------------------------
## Core Components
## --------------------------

class FeedbackManager:
    """Handles storage and retrieval of feedback data"""
    
    def __init__(self, storage_path: str = "data/feedback_data.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save_feedback(self, feedback: Feedback) -> None:
        """Save feedback to persistent storage"""
        with open(self.storage_path, "a", encoding="utf-8") as f:
            json.dump(feedback.to_dict(), f)
            f.write("\n")

    def load_feedback(self) -> List[Feedback]:
        """Load all feedback from storage"""
        feedback_data = []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    feedback_data.append(Feedback.from_dict(json.loads(line.strip())))
        except FileNotFoundError:
            print(f"No feedback data found at {self.storage_path}")
        return feedback_data


class DocumentProcessor:
    """Handles document loading and processing"""
    
    @staticmethod
    def load_pdf(path: str) -> str:
        """Load PDF content as text"""
        loader = PyPDFLoader(path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    @staticmethod
    def create_vectorstore(text: str) -> VectorStore:
        """Create a vector store from text"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)


class RelevanceScorer:
    """Handles relevance scoring of documents based on feedback"""
    
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
            template="""
            Determine if the following feedback response is relevant to the current query and document content.
            You are also provided with the Feedback original query that was used to generate the feedback response.
            Current query: {query}
            Feedback query: {feedback_query}
            Document content: {doc_content}
            Feedback response: {feedback_response}

            Is this feedback relevant? Respond with only 'Yes' or 'No'.
            """
        )
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)

    def adjust_scores(self, query: str, docs: List[Document], feedback_data: List[Feedback]) -> List[Document]:
        """Adjust document relevance scores based on feedback"""
        chain = self.prompt_template | self.llm.with_structured_output(
            ResponseModel, 
            method="function_calling"
        )

        for doc in docs:
            relevant_feedback = [
                fb for fb in feedback_data 
                if self._is_relevant(chain, query, fb, doc)
            ]

            if relevant_feedback:
                avg_relevance = sum(fb.relevance for fb in relevant_feedback) / len(relevant_feedback)
                doc.metadata['relevance_score'] = doc.metadata.get('relevance_score', 1) * (avg_relevance / 3)

        return sorted(docs, key=lambda x: x.metadata.get('relevance_score', 1), reverse=True)

    def _is_relevant(self, chain, query: str, feedback: Feedback, doc: Document) -> bool:
        """Check if feedback is relevant to the current query and document"""
        input_data = {
            "query": query,
            "feedback_query": feedback.query,
            "doc_content": doc.page_content[:1000],
            "feedback_response": feedback.response
        }
        result = chain.invoke(input_data).answer
        return result.lower() == 'yes'


## --------------------------
## Main RAG System
## --------------------------

class RetrievalAugmentedGeneration:
    """Main RAG system with feedback integration"""
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.feedback_manager = FeedbackManager()
        self.relevance_scorer = RelevanceScorer()
        
        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all RAG components"""
        try:
            # Load and process document
            self.content = DocumentProcessor.load_pdf(self.document_path)
            self.vectorstore = DocumentProcessor.create_vectorstore(self.content)
            self.retriever = self.vectorstore.as_retriever()
            
            # Initialize LLM and QA chain
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)
            self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG system: {str(e)}")

    def query(self, question: str, relevance: int = 5, quality: int = 5) -> str:
        """
        Execute a query against the RAG system and store feedback
        
        Args:
            question: The question to ask
            relevance: Relevance score for feedback (1-5)
            quality: Quality score for feedback (1-5)
            
        Returns:
            The generated answer
        """
        try:
            # Execute query
            response = self.qa_chain.invoke({"query": question})["result"]
            
            # Store feedback
            feedback = Feedback(
                query=question,
                response=response,
                relevance=relevance,
                quality=quality
            )
            self.feedback_manager.save_feedback(feedback)
            
            # Adjust relevance scores based on feedback
            docs = self.retriever.invoke(question)
            adjusted_docs = self.relevance_scorer.adjust_scores(
                question, 
                docs, 
                self.feedback_manager.load_feedback()
            )
            
            # Update retriever parameters
            self.retriever.search_kwargs.update({
                'k': len(adjusted_docs),
                'docs': adjusted_docs
            })
            
            return response
        
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")

    def fine_tune(self) -> None:
        """Fine-tune the vectorstore based on accumulated feedback"""
        try:
            feedback_data = self.feedback_manager.load_feedback()
            good_responses = [
                fb for fb in feedback_data 
                if fb.relevance >= 4 and fb.quality >= 4
            ]
            
            if good_responses:
                additional_texts = " ".join(
                    f"{fb.query} {fb.response}" for fb in good_responses
                )
                all_texts = self.content + additional_texts
                self.vectorstore = DocumentProcessor.create_vectorstore(all_texts)
                self.retriever = self.vectorstore.as_retriever()
                
        except Exception as e:
            raise RuntimeError(f"Fine-tuning failed: {str(e)}")


## --------------------------
## CLI Interface
## --------------------------

class RAGCLI:
    """Command line interface for the RAG system"""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Run the RAG system with feedback integration."
        )
        parser.add_argument(
            '--path', 
            type=str, 
            default="data/Understanding_Climate_Change.pdf",
            help="Path to the document."
        )
        parser.add_argument(
            '--query', 
            type=str, 
            default='What is the greenhouse effect?',
            help="Query to ask the RAG system."
        )
        parser.add_argument(
            '--relevance', 
            type=int, 
            default=5, 
            help="Relevance score for the feedback (1-5)."
        )
        parser.add_argument(
            '--quality', 
            type=int, 
            default=5, 
            help="Quality score for the feedback (1-5)."
        )
        return parser.parse_args()

    @staticmethod
    def run() -> None:
        """Run the RAG system from command line"""
        args = RAGCLI.parse_args()
        
        try:
            rag = RetrievalAugmentedGeneration(args.path)
            result = rag.query(args.query, args.relevance, args.quality)
            print(f"Response: {result}")
            
            # Optional: Fine-tune after query
            rag.fine_tune()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    RAGCLI.run()