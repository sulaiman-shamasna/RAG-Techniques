import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path sicnce we work with notebooks
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) 
from helper_functions import *
from evaluation.evaluate_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        
        # Create a base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create an explanation chain
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Analyze the relationship between the following query and the retrieved context.
            Explain why this context is relevant to the query and how it might help answer the query.
            
            Query: {query}
            
            Context: {context}
            
            Explanation:
            """
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        explained_results = []
        
        for doc in docs:
            # Generate explanation
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
        
        return explained_results
    
# Usage
texts = [
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Water expands when it freezes, which is why ice floats on liquid water.",
    "Earth's gravity causes objects to fall at a rate of 9.8 meters per second squared.",
    "DNA carries genetic information and is found in nearly all living organisms.",
    "Sound travels faster in water than in air due to the density of the medium."
]

explainable_retriever = ExplainableRetriever(texts)


query = "What's the speed of light?"
results = explainable_retriever.retrieve_and_explain(query)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {result['content']}")
    print(f"Explanation: {result['explanation']}")
    print()