from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
1 - Query Rewriting: Reformulating queries to improve retrieval.
"""

re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

# Create a prompt template for query rewriting
query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

# Create an LLMChain for query rewriting
query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.
    
    Args:
    original_query (str): The original user query
    
    Returns:
    str: The rewritten query
    """
    response = query_rewriter.invoke(original_query)
    return response.content
