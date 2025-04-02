import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class DocumentGrader(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class RAGPipeline:
    def __init__(self, embedding_model: Any, llm_model: str = "gpt-4", temperature: float = 0):
        """
        Initialize the RAG pipeline with embedding and language models.
        
        Args:
            embedding_model: The embedding model to use
            llm_model: Name of the LLM to use for generation
            temperature: Temperature parameter for LLM
        """
        self.embedding_model = embedding_model
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
        self.vectorstore = None
        self.retriever = None
        
    def load_and_split_documents(self, urls: List[str], chunk_size: int = 500, chunk_overlap: int = 0) -> List[Document]:
        """
        Load documents from URLs and split them into chunks.
        
        Args:
            urls: List of URLs to load documents from
            chunk_size: Size of each document chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
        """
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs_list)
    
    def create_vectorstore(self, documents: List[Document], collection_name: str = "rag") -> None:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to add to vector store
            collection_name: Name of the collection in the vector store
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4},
        )
    
    def retrieve_documents(self, question: str) -> List[Document]:
        """
        Retrieve documents relevant to a question.
        
        Args:
            question: The question to retrieve documents for
            
        Returns:
            List of retrieved documents
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call create_vectorstore() first.")
        return self.retriever.invoke(question)
    
    def grade_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Grade documents for relevance to a question.
        
        Args:
            question: The question to grade against
            documents: List of documents to grade
            
        Returns:
            List of relevant documents
        """
        # Set up grader
        structured_llm_grader = self.llm.with_structured_output(DocumentGrader)
        
        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        
        retrieval_grader = grade_prompt | structured_llm_grader
        
        # Filter documents
        relevant_docs = []
        for doc in documents:
            result = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if result.binary_score == "yes":
                relevant_docs.append(doc)
                
        return relevant_docs
    
    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """
        Generate an answer to a question using relevant documents.
        
        Args:
            question: The question to answer
            documents: List of relevant documents
            
        Returns:
            Generated answer
        """
        # Format documents
        def format_docs(docs: List[Document]) -> str:
            return "\n".join(
                f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" 
                for i, doc in enumerate(docs)
            )
        
        # Set up prompt and chain
        system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
        Use three-to-five sentences maximum and keep the answer concise."""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
            ]
        )
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        # Generate answer
        return rag_chain.invoke({
            "documents": format_docs(documents), 
            "question": question
        })

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Pipeline for Question Answering")
    parser.add_argument(
        "--question", 
        type=str, 
        default="what are the different kinds of agentic design patterns?",
        help="The question to answer"
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default="gpt-4",
        help="The LLM model to use for generation"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0,
        help="Temperature parameter for LLM generation"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=500,
        help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=0,
        help="Overlap between document chunks"
    )
    return parser.parse_args()

def main():
    # Load environment variables
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    
    # Parse arguments
    args = parse_arguments()
    
    # URLs to process
    urls = [
        "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
        "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
    ]
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        embedding_model=OpenAIEmbeddings(),
        llm_model=args.llm_model,
        temperature=args.temperature
    )
    
    # Load and process documents
    doc_splits = rag_pipeline.load_and_split_documents(
        urls, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap
    )
    rag_pipeline.create_vectorstore(doc_splits)
    
    # Retrieve and process documents
    retrieved_docs = rag_pipeline.retrieve_documents(args.question)
    relevant_docs = rag_pipeline.grade_documents(args.question, retrieved_docs)
    
    # Display first relevant document
    if relevant_docs:
        print(f"Title: {relevant_docs[0].metadata['title']}\n\nSource: {relevant_docs[0].metadata['source']}\n\nContent: {relevant_docs[0].page_content}\n")
        print(10 * "=")
    
    # Generate and display answer
    answer = rag_pipeline.generate_answer(args.question, relevant_docs)
    print(answer)

if __name__ == "__main__":
    main()