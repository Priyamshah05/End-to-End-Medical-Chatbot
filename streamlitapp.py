import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    st.success("FAISS database loaded successfully.")
except Exception as e:
    st.error(f"Error loading FAISS database: {e}")

st.title("Medical Chatbot")
st.write("Ask me anything about medical topics!")

question = st.text_input("Your question:")
if st.button("Ask"):
    try:
        response = db.search(question, search_type='similarity')
        st.markdown(f"**Question:** {question}\n\n**Answer:** {response[0].page_content}")
    except Exception as e:
        st.error(f"Error: {e}")
