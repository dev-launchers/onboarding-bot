# Import necessary modules and classes from the Langchain library and other dependencies
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache
import chainlit as cl
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Access the OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_KEY")

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



def set_custom_prompt():
    """
    Function to set a custom prompt template for QA retrieval
    """
    
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

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
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Chainlit code
@cl.on_chat_start
async def start():
    # Initialize the QA bot and start a conversation
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    # Get the QA bot chain from the user session
    chain = cl.user_session.get("chain") 
    cbh = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cbh.answer_reached = True
    # Send the user's message to the QA bot chain
    res = await chain.acall(message, callbacks=[cbh])
    answer = res["result"]
    sources = res["source_documents"]
    # answer += "\n \n \n  No sources found"

    await cl.Message(content=answer).send()
    
