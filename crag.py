import os
import sys
import json
from typing import List, Tuple
from dotenv import load_dotenv

from pydantic import BaseModel, Field  # Pydantic v2-compatible
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# Setup path and environment
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evaluate_rag import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize components
path = "data/Understanding_Climate_Change.pdf"
vectorstore = encode_pdf(path)
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
search = DuckDuckGoSearchResults()

# -------------------------- Retrieval Evaluator --------------------------

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="Relevance score between 0 and 1.")

def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate.from_template(
        "On a scale from 0 to 1, how relevant is the following document to the query?\n"
        "Query: {query}\nDocument: {document}\nRelevance score:"
    )
    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)
    return chain.invoke({"query": query, "document": document}).relevance_score

# -------------------------- Knowledge Refinement --------------------------

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="Bullet points of key information.")

def knowledge_refinement(document: str) -> List[str]:
    prompt = PromptTemplate.from_template(
        "Extract the key information from the following document in bullet points:\n{document}\nKey points:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    result = chain.invoke({"document": document}).key_points
    return [pt.strip() for pt in result.split('\n') if pt.strip()]

# -------------------------- Query Rewriting --------------------------

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="Rewritten web search query.")

def rewrite_query(query: str) -> str:
    prompt = PromptTemplate.from_template(
        "Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    return chain.invoke({"query": query}).query.strip()

# -------------------------- Web Search + Helpers --------------------------

def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    try:
        results = json.loads(results_string)
        return [(r.get("title", "Untitled"), r.get("link", "")) for r in results]
    except json.JSONDecodeError:
        print("Error parsing search results.")
        return []

def perform_web_search(query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources

# -------------------------- Retrieval + Evaluation --------------------------

def retrieve_documents(query: str, faiss_index, k: int = 3) -> List[str]:
    docs = faiss_index.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    return [retrieval_evaluator(query, doc) for doc in documents]

# -------------------------- Response Generation --------------------------

def generate_response(query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
    prompt = PromptTemplate.from_template(
        "Based on the following knowledge, answer the query. Include the sources with their links (if available) at the end of your answer:\n"
        "Query: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
    )
    input_vars = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    return (prompt | llm).invoke(input_vars).content

# -------------------------- CRAG Process --------------------------

def crag_process(query: str, faiss_index) -> str:
    print(f"\nProcessing query: {query}")
    retrieved_docs = retrieve_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)

    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print(f"Evaluation scores: {eval_scores}")

    max_score = max(eval_scores) if eval_scores else 0
    sources = []

    if max_score > 0.7:
        print("Action: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        print("Action: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        print("Action: Ambiguous - Combining retrieved and web knowledge")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        retrieved_knowledge = knowledge_refinement(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    print("\nFinal knowledge:\n", final_knowledge)
    print("\nSources:")
    for title, link in sources:
        print(f"{title}: {link}" if link else title)

    response = generate_response(query, final_knowledge, sources)
    print("\nResponse generated")
    return response

# -------------------------- Run Sample Queries --------------------------

if __name__ == "__main__":
    for query in [
        "What are the main causes of climate change?",
        "how did harry beat quirrell?"
    ]:
        answer = crag_process(query, vectorstore)
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
