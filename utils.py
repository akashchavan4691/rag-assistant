import os
from langchain_chroma import Chroma
from chromadb.utils import embedding_functions

DB_DIR = "./chroma_db"

def get_vector_store():
    # ChromaDB ka apna built-in embedding function use karte hain
    # Ye ONNX use karta hai, koi API key ya torch nahi chahiye!
    # Pehli baar ~23MB model download hoga
    chroma_ef = embedding_functions.DefaultEmbeddingFunction()

    # LangChain wrapper ke liye ek custom class
    class ChromaEmbeddingWrapper:
        def embed_documents(self, texts):
            return chroma_ef(texts)

        def embed_query(self, text):
            return chroma_ef([text])[0]

    embeddings = ChromaEmbeddingWrapper()

    vector_store = Chroma(
        collection_name="my_documents",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store
