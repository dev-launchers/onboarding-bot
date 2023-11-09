from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
# Load the environment variables from the .env file
load_dotenv()

# Access the API key using os.environ
openai_api_key = os.environ.get("OPENAI_KEY")
OPENAI_API_KEY= os.environ.get("OPENAI_KEY")

DB_FAISS_PATH = 'vectorstore/db_faiss'

app = Flask(__name__)

custom_prompt_template = """ You are an experience customer service agent working for EQ Bank.
Answer the questions politely.
Use the following pieces of information to answer the user's question.
You have access to the folowing knowledgebase, with the information on 
1) EQ Bank Account
2) Bank Card
3) tax free savings account

If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    # llm = CTransformers( model = "llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens = 512, temperature = 0.5)
    return llm

#QA Model Function
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cbh = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cbh.answer_reached = True
    res = await chain.acall(message, callbacks=[cbh])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n \n \n Sources:" + str(sources) + "\n \n \n"
    else:
        answer += "\n \n \n  No sources found"

    await cl.Message(content=answer).send()
