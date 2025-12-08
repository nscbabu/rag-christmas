# ingest.py

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
PERSIST_DIR = "./chroma_db_resume"  # Where the vector store will be saved
RESUME_FILE = "sample_resume.txt"
EMBEDDING_MODEL = "nomic-embed-text" # Using the model pulled in Module 2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

print(f"--- Starting Data Ingestion for {RESUME_FILE} ---")

# 1. Load the document
try:
    loader = TextLoader(RESUME_FILE)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
except FileNotFoundError:
    print(f"Error: {RESUME_FILE} not found. Please check Module 1.")
    exit()

# 2. Split the document into manageable chunks (for effective retrieval)
print(f"Splitting document into chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks.")

# 3. Initialize Embeddings and Vector Store
print(f"Initializing Ollama Embedding Model: {EMBEDDING_MODEL}...")
# NOTE: The OllamaEmbeddings class automatically communicates with the local Ollama server.
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# 4. Create or Load the Vector Store (The RAG Database)
print(f"Creating/Updating ChromaDB at: {PERSIST_DIR}...")
# This step calculates the embeddings and saves them to the disk.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
vectorstore.persist()

print("--- Ingestion Complete! Vector store is ready. ---")
