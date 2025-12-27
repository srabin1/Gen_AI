import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# --- Keys ---
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")  # optional for public models; helpful to avoid rate limits

if not groq_key:
    st.error("GROQ_API_KEY is missing. Put it in your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_key
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- LLM (still Groq) ---
llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based only on the provided context.
Provide the most accurate response.

<context>
{context}
</context>

Question: {input}
"""
)

st.title("RAG Document Q&A With HuggingFace Embeddings")

# ---- Build vector DB once (cached) ----
@st.cache_resource(show_spinner=True)
def build_vector_db(pdf_folder: str):
    # Hugging Face embeddings (local + lightweight)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    loader = PyPDFDirectoryLoader(pdf_folder)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)  # remove [:50] unless you want to limit

    vectors = FAISS.from_documents(chunks, embeddings)
    return vectors

# ---- UI ----
st.write("1) Click **Document Embedding** once to build the vector DB.")
st.write("2) Then ask questions.")

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if st.button("Document Embedding"):
    try:
        st.session_state.vectors = build_vector_db("research_papers")
        st.success("Vector Database is ready ✅")
    except Exception as e:
        st.error(f"Failed to build vector DB: {e}")
        st.stop()

with st.form("qa_form"):
    user_prompt = st.text_input("Enter your query from the research paper(s)")
    ask = st.form_submit_button("Ask")

if ask:
    if st.session_state.vectors is None:
        st.warning("Please click **Document Embedding** first to create the vector database.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.caption(f"Response time: {time.process_time() - start:.3f} seconds")

    st.write(response.get("answer", ""))

    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response.get("context", []), start=1):
            st.write(f"Chunk {i}:")
            st.write(doc.page_content)
            st.write("---")
