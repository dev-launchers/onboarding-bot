# Import necessary modules and classes from the Langchain library and other dependencies
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

from flask import Flask, request
from flask_cors import CORS

import time

# Load environment variables from a .env file
load_dotenv()

# Access the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Path to the FAISS vectorstore database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template for the QA bot
custom_prompt_template = """You are a helpful customer personal assistant agent working for Dev Launchers organization.
your goal is to understand the customer's needs and help them. 
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

#Prompt Template initiated
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])

# Function to create a Retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Function to load the chat model
def load_llm():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    return llm

# Function to set up the QA bot
def qa_bot():
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = prompt #set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa
#############################################
chain = qa_bot()
app = Flask(__name__)

@app.route('/question', methods=['POST','GET'])
def question():
    chat_input = request.form.get("string")
    response = chain(chat_input)
    return (response["result"])

@app.route('/testQuestion', methods=['POST', 'GET'])
def testQuestion():
    chat_input = request.form.get("string")
    print(chat_input)
    test = ("text: \"Test\"")
    time.sleep(3)
    return chat_input

CORS(app)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
