import os
from dotenv import load_dotenv

# Load environment variables from '.env' file
load_dotenv()

# Set only OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()

# Docs to index
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag",
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4},
)

# Prompt user for a question
user_question = input("Enter your question (or press Enter to use the default question): ")
question = user_question.strip() or "what are the different kinds of agentic design patterns?"

# Retrieve docs
docs = retriever.invoke(question)

# Display the first retrieved document
print(f"Title: {docs[0].metadata['title']}\n\nSource: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")
print(10 * "=")

