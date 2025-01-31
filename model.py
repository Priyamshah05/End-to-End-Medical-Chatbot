from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS database loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS database: {e}")

@cl.on_message
async def main(message):
    try:
        response = db.search(message.content, search_type='similarity')
        formatted_response = "\n\n".join([doc.page_content for doc in response])
        await cl.Message(
            content=f"**Question:** {message.content}\n\n**Answer:** {formatted_response}"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"Error: {e}"
        ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! What can I help you with today?"
    ).send()
    
#Complete file with greeting prompt