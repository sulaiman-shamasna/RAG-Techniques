import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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
import argparse
from dotenv import load_dotenv

# Add parent directory to path for notebook compatibility
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from helper_functions import *
from evaluation.evaluate_rag import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class RaptorConfig:
    """Configuration class for RAPTOR parameters."""
    max_levels: int = 3
    n_clusters: int = 10
    retrieval_k: int = 3
    llm_model: str = "gpt-4"
    default_question: str = "What is the greenhouse effect?"


@dataclass
class RetrievedDocument:
    """Dataclass to store information about retrieved documents."""
    index: int
    content: str
    metadata: Dict[str, Any]
    level: int
    similarity_score: float


@dataclass
class QueryResult:
    """Dataclass to store the complete result of a RAPTOR query."""
    query: str
    retrieved_documents: List[RetrievedDocument]
    num_docs_retrieved: int
    context_used: str
    answer: str
    model_used: str


class RaptorTreeBuilder:
    """Class responsible for building the RAPTOR tree structure."""
    
    def __init__(self, config: RaptorConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=config.llm_model)
    
    def extract_text(self, item: Any) -> str:
        """Extract text content from either a string or an AIMessage object.
        
        Args:
            item: Input item which could be a string or AIMessage
            
        Returns:
            Extracted text content
        """
        if isinstance(item, AIMessage):
            return item.content
        return str(item)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAIEmbeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        logging.info(f"Embedding {len(texts)} texts")
        return self.embeddings.embed_documents([self.extract_text(text) for text in texts])
    
    def perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform clustering on embeddings using Gaussian Mixture Model.
        
        Args:
            embeddings: Array of embeddings to cluster
            
        Returns:
            Array of cluster labels
        """
        n_clusters = min(self.config.n_clusters, len(embeddings) // 2)
        logging.info(f"Performing clustering with {n_clusters} clusters")
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gm.fit_predict(embeddings)
    
    def summarize_texts(self, texts: List[str]) -> str:
        """Summarize a list of texts using OpenAI.
        
        Args:
            texts: List of texts to summarize
            
        Returns:
            Generated summary
        """
        logging.info(f"Summarizing {len(texts)} texts")
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following text concisely:\n\n{text}"
        )
        chain = prompt | self.llm
        input_data = {"text": texts}
        return chain.invoke(input_data)
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, level: int):
        """Visualize clusters using PCA.
        
        Args:
            embeddings: Array of embeddings
            labels: Cluster labels
            level: Current tree level for title
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Cluster Visualization - Level {level}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()
    
    def build_raptor_tree(self, texts: List[str]) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships.
        
        Args:
            texts: List of input texts to build the tree from
            
        Returns:
            Dictionary mapping levels to DataFrames containing tree information
        """
        results = {}
        current_texts = [self.extract_text(text) for text in texts]
        current_metadata = [{"level": 0, "origin": "original", "parent_id": None, "id": f"doc_{i}"} 
                           for i in range(len(texts))]
        
        for level in range(1, self.config.max_levels + 1):
            logging.info(f"Processing level {level}")
            
            embeddings = self.embed_texts(current_texts)
            cluster_labels = self.perform_clustering(np.array(embeddings))
            
            df = pd.DataFrame({
                'text': current_texts,
                'embedding': embeddings,
                'cluster': cluster_labels,
                'metadata': current_metadata
            })
            
            results[level-1] = df
            
            summaries = []
            new_metadata = []
            for cluster in df['cluster'].unique():
                cluster_docs = df[df['cluster'] == cluster]
                cluster_texts = cluster_docs['text'].tolist()
                cluster_metadata = cluster_docs['metadata'].tolist()
                summary = self.summarize_texts(cluster_texts)
                summaries.append(summary)
                new_metadata.append({
                    "level": level,
                    "origin": f"summary_of_cluster_{cluster}_level_{level-1}",
                    "child_ids": [meta.get('id') for meta in cluster_metadata],
                    "id": f"summary_{level}_{cluster}"
                })
            
            current_texts = summaries
            current_metadata = new_metadata
            
            if len(current_texts) <= 1:
                results[level] = pd.DataFrame({
                    'text': current_texts,
                    'embedding': self.embed_texts(current_texts),
                    'cluster': [0],
                    'metadata': current_metadata
                })
                logging.info(f"Stopping at level {level} as we have only one summary")
                break
        
        return results


class RaptorRetriever:
    """Class responsible for retrieval operations in the RAPTOR system."""
    
    def __init__(self, config: RaptorConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=config.llm_model)
    
    def build_vectorstore(self, tree_results: Dict[int, pd.DataFrame]) -> FAISS:
        """Build a FAISS vectorstore from all texts in the RAPTOR tree.
        
        Args:
            tree_results: Dictionary of tree level DataFrames
            
        Returns:
            FAISS vectorstore containing all documents
        """
        all_texts = []
        all_embeddings = []
        all_metadatas = []
        
        for level, df in tree_results.items():
            all_texts.extend([str(text) for text in df['text'].tolist()])
            all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding 
                                 for embedding in df['embedding'].tolist()])
            all_metadatas.extend(df['metadata'].tolist())
        
        logging.info(f"Building vectorstore with {len(all_texts)} texts")
        
        documents = [Document(page_content=str(text), metadata=metadata) 
                     for text, metadata in zip(all_texts, all_metadatas)]
        
        return FAISS.from_documents(documents, self.embeddings)
    
    def create_retriever(self, vectorstore: FAISS) -> ContextualCompressionRetriever:
        """Create a retriever with contextual compression.
        
        Args:
            vectorstore: FAISS vectorstore to use as base retriever
            
        Returns:
            Contextual compression retriever
        """
        logging.info("Creating contextual compression retriever")
        base_retriever = vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_template(
            "Given the following context and question, extract only the relevant information for answering the question:\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Relevant Information:"
        )
        
        extractor = LLMChainExtractor.from_llm(self.llm, prompt=prompt)
        
        return ContextualCompressionRetriever(
            base_compressor=extractor,
            base_retriever=base_retriever
        )
    
    def hierarchical_retrieval(self, query: str, retriever: ContextualCompressionRetriever, 
                              max_level: int) -> List[Document]:
        """Perform hierarchical retrieval starting from the highest level.
        
        Args:
            query: Search query
            retriever: Configured retriever
            max_level: Maximum tree level to search
            
        Returns:
            List of retrieved documents
        """
        all_retrieved_docs = []
        
        for level in range(max_level, -1, -1):
            level_docs = retriever.get_relevant_documents(
                query,
                filter=lambda meta: meta['level'] == level
            )
            all_retrieved_docs.extend(level_docs)
            
            if level_docs and level > 0:
                child_ids = [doc.metadata.get('child_ids', []) for doc in level_docs]
                child_ids = [item for sublist in child_ids for item in sublist if item is not None]
                
                if child_ids:
                    child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"
                    query += child_query
        
        return all_retrieved_docs
    
    def raptor_query(self, query: str, retriever: ContextualCompressionRetriever, 
                    max_level: int) -> QueryResult:
        """Process a query using the RAPTOR system with hierarchical retrieval.
        
        Args:
            query: Search query
            retriever: Configured retriever
            max_level: Maximum tree level to search
            
        Returns:
            QueryResult object containing all query information
        """
        logging.info(f"Processing query: {query}")
        
        relevant_docs = self.hierarchical_retrieval(query, retriever, max_level)
        
        doc_details = []
        for i, doc in enumerate(relevant_docs, 1):
            doc_details.append(RetrievedDocument(
                index=i,
                content=doc.page_content,
                metadata=doc.metadata,
                level=doc.metadata.get('level', -1),
                similarity_score=doc.metadata.get('score', 0.0)
            ))
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = ChatPromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)
        
        logging.info("Query processing completed")
        
        return QueryResult(
            query=query,
            retrieved_documents=doc_details,
            num_docs_retrieved=len(relevant_docs),
            context_used=context,
            answer=answer,
            model_used=self.llm.model_name,
        )


def print_query_details(result: QueryResult):
    """Print detailed information about the query process.
    
    Args:
        result: QueryResult object containing query information
    """
    print(f"\n{'='*50}")
    print(f"Query: {result.query}")
    print(f"\nNumber of documents retrieved: {result.num_docs_retrieved}")
    print(f"\nRetrieved Documents:")
    
    for doc in result.retrieved_documents:
        print(f"\n  Document {doc.index}:")
        print(f"    Content: {doc.content[:100]}...")
        print(f"    Similarity Score: {doc.similarity_score:.4f}")
        print(f"    Tree Level: {doc.level}")
        print(f"    Origin: {doc.metadata.get('origin', 'Unknown')}")
        if 'child_ids' in doc.metadata:
            print(f"    Number of Child Documents: {len(doc.metadata['child_ids'])}")
    
    print(f"\nContext used for answer generation:")
    print(result.context_used[:500] + "..." if len(result.context_used) > 500 else result.context_used)
    
    print(f"\nGenerated Answer:")
    print(result.answer)
    
    print(f"\nModel Used: {result.model_used}")
    print(f"{'='*50}\n")


def load_documents(file_path: str) -> List[Document]:
    """Load documents from a PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of loaded documents
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def get_user_question(default_question: str) -> str:
    """Prompt user for question or use default if empty input.
    
    Args:
        default_question: Default question to use if user doesn't provide one
        
    Returns:
        The question to use for querying
    """
    user_input = input(f"Enter your question (or press Enter to use default: '{default_question}'): ")
    return user_input.strip() if user_input.strip() else default_question


def main(config: RaptorConfig, file_path: str):
    """Main execution function for RAPTOR system.
    
    Args:
        config: RAPTOR configuration
        file_path: Path to input PDF file
    """
    documents = load_documents(file_path)
    texts = [doc.page_content for doc in documents]
    
    # Build RAPTOR tree
    tree_builder = RaptorTreeBuilder(config)
    tree_results = tree_builder.build_raptor_tree(texts)
    
    # Set up retrieval system
    retriever = RaptorRetriever(config)
    vectorstore = retriever.build_vectorstore(tree_results)
    compression_retriever = retriever.create_retriever(vectorstore)
    
    question = get_user_question(config.default_question)
    
    result = retriever.raptor_query(question, compression_retriever, config.max_levels)
    print_query_details(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval")
    parser.add_argument("--file", type=str, default="data/Understanding_Climate_Change.pdf",
                       help="Path to the input PDF file")
    parser.add_argument("--max_levels", type=int, default=3,
                       help="Maximum levels for the RAPTOR tree")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="Number of clusters for each level")
    parser.add_argument("--retrieval_k", type=int, default=3,
                       help="Number of documents to retrieve at each level")
    parser.add_argument("--llm_model", type=str, default="gpt-4",
                       help="LLM model to use for summarization and question answering")
    parser.add_argument("--default_question", type=str, 
                       default="What is the greenhouse effect?",
                       help="Default question to use if none is provided")
    
    args = parser.parse_args()
    
    config = RaptorConfig(
        max_levels=args.max_levels,
        n_clusters=args.n_clusters,
        retrieval_k=args.retrieval_k,
        llm_model=args.llm_model,
        default_question=args.default_question
    )
    
    main(config, args.file)