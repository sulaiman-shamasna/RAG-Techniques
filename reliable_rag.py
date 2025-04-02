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



"""
Script Execution Example:
-------------------------

Output:

Enter your question (or press Enter to use the default question): 
Title: Agentic Design Patterns Part 5, Multi-Agent Collaboration

Source: https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io

Content: scalable and highly secure code. By decomposing the overall task into subtasks, we can optimize the subtasks better.Perhaps most important, the multi-agent design pattern gives us, as developers, a framework for breaking down complex tasks into subtasks. When writing code to run on a single CPU, we often break our program up into different processes or threads. This is a useful abstraction that lets us decompose a task, like implementing a web browser, into subtasks that are easier to code. I find thinking through multi-agent roles to be a useful abstraction as well.In many companies, managers routinely decide what roles to hire, and then how to split complex projects — like writing a large piece of software or preparing a research report — into smaller tasks to assign to employees with different specialties. Using multiple agents is analogous. Each agent implements its own workflow, has its own memory (itself a rapidly evolving area in agentic technology: how can an agent remember enough of its past interactions to perform better on upcoming ones?), and may ask other agents for help. Agents can also engage in Planning and Tool Use. This results in a cacophony of LLM calls and message passing between agents that can result in very complex workflows. While managing people is hard, it's a sufficiently familiar idea that it gives us a mental framework for how to "hire" and assign tasks to our AI agents. Fortunately, the damage from mismanaging an AI agent is much lower than that from mismanaging humans! Emerging frameworks like AutoGen, Crew AI, and LangGraph, provide rich ways to build multi-agent solutions to problems. If you're interested in playing with a fun multi-agent system, also check out ChatDev, an open source implementation of a set of agents that run a virtual software company. I encourage you to check out their GitHub repo and perhaps clone the repo and run the system yourself. While it may not always produce what you want, you might be amazed at how well it does. Like the design pattern of Planning, I find the output quality of multi-agent collaboration hard to predict, especially when allowing agents to interact freely and providing them with multiple tools. The more mature patterns of Reflection and Tool Use are more reliable. I hope you enjoy playing with these agentic design patterns and that they produce amazing results for you! If you're interested in learning more, I ... 


Answer: 

Agentic design patterns are strategies for building AI agents. They include:

1. Reflection: The AI agent examines its own work to find ways to improve it.
2. Tool Use: The AI agent is given tools such as web search, code execution, or other functions to help it gather information, take action, or process data.
3. Planning: The AI agent comes up with and executes a multistep plan to achieve a goal.
4. Multi-agent Collaboration: More than one AI agent work together, splitting up tasks and discussing and debating ideas, to come up with better solutions than a single agent would.
"""