import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evaluate_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class categories_options(BaseModel):
        category: str = Field(description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual", example="Factual")


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)


    def classify(self, query):
        print("clasiffying query")
        return self.chain.invoke(query).category
    
class categories_options(BaseModel):
        category: str = Field(description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual", example="Factual")


class QueryClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)


    def classify(self, query):
        print("clasiffying query")
        return self.chain.invoke(query).category
    
class relevant_score(BaseModel):
        score: float = Field(description="The relevance score of the document to the query", example=8.0)

class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving factual")
        # Use LLM to enhance the query
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'enhande query: {enhanced_query}')

        # Retrieve documents using the enhanced query
        docs = self.db.similarity_search(enhanced_query, k=k*2)

        # Use LLM to rank the relevance of retrieved documents
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        print("ranking docs")
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]
    
class SelectedIndices(BaseModel):
    indices: List[int] = Field(description="Indices of selected documents", example=[0, 1, 2, 3])

class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description="List of sub-queries for comprehensive analysis", example=["What is the population of New York?", "What is the GDP of New York?"])

class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving analytical")
        # Use LLM to generate sub-queries for comprehensive analysis
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )

        llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        sub_queries_chain = sub_queries_prompt | llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'sub queries for comprehensive analysis: {sub_queries}')

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Use LLM to ensure diversity and relevance
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\n
            Return only the indices of selected documents as a list of integers."""
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data).indices
        print(f'selected diverse and relevant documents')
        
        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]