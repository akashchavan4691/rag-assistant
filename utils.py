import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_db"

def get_vector_store():
    # HuggingFace Inference API - free, no heavy torch needed on server
    hf_token = os.environ.get("HF_TOKEN")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        collection_name="my_documents",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store
