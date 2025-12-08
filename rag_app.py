# rag_app.py

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
# --- Configuration ---
PERSIST_DIR = "./chroma_db_resume"
LLM_MODEL = "llama3" # Using the model pulled in Module 2
EMBEDDING_MODEL = "nomic-embed-text" 

# --- Retrieval-Augmented Generation Prompt Template ---
# This is the "A" (Augmentation) part of RAG. It guides the LLM.
# We are instructing the LLM to only use the provided context.
RAG_PROMPT_TEMPLATE = """
You are an expert AI assistant tasked with answering questions based ONLY on the provided context,
which is the resume of an employee named Johnathan Thompson.

If the answer cannot be found within the context, you MUST state "I cannot find this information in Johnathan Thompson's resume."
Do not make up any information. Use a friendly and professional tone.

Context:
---
{context}
---
Question: {question}
Answer:
"""

def run_rag_app():
    print("--- Loading RAG Application ---")
    
    # 1. Initialize Embeddings and Vector Store
    # We load the existing vector store created by ingest.py
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    # 2. Setup Retriever
    # The retriever finds the top 3 most relevant chunks from the resume.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

    # 3. Initialize Local LLM (Llama 3)
    llm = Ollama(model=LLM_MODEL, temperature=0.1)
    
    # 4. Create the LangChain Prompt
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # 5. Create the RAG Chain (The full pipeline: Question -> Retrieve -> Augment -> Generate)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "Stuff" all retrieved documents into the context.
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt}
    )

    print(f"RAG App Ready (LLM: {LLM_MODEL}, Embedding: {EMBEDDING_MODEL}, Context Chunks: 3)")
    print("Type 'exit' to quit the application.")
    
    # --- Main Application Loop ---
    while True:
        question = input("\nAsk about the resume > ")
        if question.lower() == 'exit':
            break
        
        if not question:
            continue

        print("\n[Thinking...]")
        
        # Invoke the RAG Chain
        result = qa_chain.invoke({"query": question})
        
        # Display Results
        print("\n--- LLM Answer ---")
        print(result["result"])
        print("\n--- Source Documents (Context Used) ---")
        
        # Show which chunks were used to generate the answer
        for i, doc in enumerate(result["source_documents"]):
            print(f"Source {i+1} (Score: {doc.metadata.get('score', 'N/A')}):")
            print(f"  Content snippet: \"{doc.page_content[:150]}...\"")
        print("------------------")

if __name__ == "__main__":
    run_rag_app()