import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.mixture import GaussianMixture
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import AIMessage
from langchain.docstore.document import Document

import matplotlib.pyplot as plt
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evaluate_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini")

def extract_text(item):
    """Extract text content from either a string or an AIMessage object."""
    if isinstance(item, AIMessage):
        return item.content
    return item

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using OpenAIEmbeddings."""
    logging.info(f"Embedding {len(texts)} texts")
    return embeddings.embed_documents([extract_text(text) for text in texts])

