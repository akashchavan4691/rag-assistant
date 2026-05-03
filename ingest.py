import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_vector_store
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

def load_excel(file_path):
    # Excel ko load karne ka aasan tareeqa pandas se
    df = pd.read_excel(file_path)
    # Har row ko ek document bana dete hain
    docs = []
    for index, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        docs.append(Document(page_content=row_text, metadata={"source": file_path, "row": index}))
    return docs

def add_document(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' nahi mili.")
        return

    ext = os.path.splitext(file_path)[-1].lower()
    docs = []

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    elif ext in [".xlsx", ".xls"]:
        docs = load_excel(file_path)
    else:
        print(f"Error: Format '{ext}' supported nahi hai.")
        return
    
    # Text Splitting (Bade documents ko chhote chunks mein todna)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Vector DB mein save karna
    vector_store = get_vector_store()
    vector_store.add_documents(splits)
    print(f"\nSuccess! Added {len(splits)} chunks from '{file_path}' to the RAG system!")

if __name__ == "__main__":
    file_path = input("Enter the path of PDF, Word, or Excel file to add: ")
    add_document(file_path)
