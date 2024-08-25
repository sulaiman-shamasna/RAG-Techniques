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