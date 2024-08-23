import sys
import os
import re
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from enum import Enum
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks

from helper_functions import *


class QuestionGeneration(Enum):
    """
    Enum class to specify the level of question generation for document processing.

    Attributes:
        DOCUMENT_LEVEL (int): Represents question generation at the entire document level.
        FRAGMENT_LEVEL (int): Represents question generation at the individual text fragment level.
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2

#Depending on the model, for Mitral 7B it can be max 8000, for Llama 3.1 8B 128k
DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100

#Embeddings and text similarity calculated on shorter texts
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16

#Questions generated on document or fragment level
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
#how many questions will be generated for specific document or fragment
QUESTIONS_PER_DOCUMENT = 40