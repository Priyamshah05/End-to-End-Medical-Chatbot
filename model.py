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
        await cl.Message(
            content=f"**Question:** {message.content}\n\n**Answer:** {response}"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"Error: {e}"
        ).send()