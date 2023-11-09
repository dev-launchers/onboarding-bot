from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the API key using os.environ
openai_api_key = os.environ.get("OPENAI_KEY")
OPENAI_API_KEY= os.environ.get("OPENAI_KEY")

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

