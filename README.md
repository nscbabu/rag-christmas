# rag-christmas
A simple RAG application to test its abilities and shortcomings.

Description
This project is a basic application to learn the basics of RAG. In this project a developer can query a text resume stored in a vector db and send it to a locally installed LLM. This project uses ChromaDB as its vector database. This project uses a text resume that is about 5 pages long, provides steps to install a vector database and open-source LLM Llama 3 8B locally. It also uses an embedding model - nomic-embed-text.

Setup instructions

Local LLM Setup with Ollama
------------------------------

1. Go to this url to download and install Ollama for your operating system https://ollama.com/download

2. Once installed go to your terminal/command prompt and try the following command and if it lists the version number we are good to go : ollama --version
Note: Ollama generally runs at http://localhost:11434/v1 you can check in your browser.

3. Now install teh Llama 3 8B by typing the following command in your terminal/command prompt: ollama run llama3

4. Once installed it will start a chat, you can test it by saying "Hello" at the prompt. To exit the chat type /bye.

Download the embedding model - nomic-embed-text
-------------------------------------------------

At your terminal or command prompt type the following command: ollama pull nomic-embed-text


Install LangChain, the Ollama connector, and ChromaDB
----------------------------------------------------------
a)At your terminal or command prompt type the following commands one by one:
python3 -m venv venv
source venv/bin/activate 
pip install langchain langchain-community langchain-chroma langchain-classic

b)To ingest the sample_resume.txt into ChromaDB we installed above run the following command:
python ingest.py

c)Now that everything is setup you can run the sample rag application by typing the following command:
python rag_app.py



