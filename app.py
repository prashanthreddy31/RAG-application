import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# ---------------- LOAD SECRETS ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .agent-card {
        background-color: #000000;
        color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #000000 0%, #000000 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .execution-step {
        background-color: #000000;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #17a2b8;
    }
    
    .source-item {
        background-color: #000000;
        color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RSSA Instruments Guide", layout="wide")
st.title("🌾 RSSA Field Instruments Guide")

st.markdown("""
This app allows you to **ask questions** related to the Field Instruments.  
It uses **RAG (Retrieval-Augmented Generation)** powered by **Groq** and **HuggingFace embeddings**.
""")

# ---------------- CACHED RESOURCES ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b"
    )

vectors = load_vectorstore()
llm = load_llm()

prompt=ChatPromptTemplate.from_template(
"""
You are a scientific assistant specializing in Instrumentation.
You will answer based only on the retrieved context.

Instructions:
- Use only the provided context.
- If mathematical equations are present, format them strictly in LaTeX.
- Use $...$ for inline equations.
- Use $$...$$ for block equations.
- Do NOT use markdown code blocks for equations.
- Do not escape backslashes.
- Do not summarize mathematical expressions.
- If not found in context, clearly say it is not available.

<context>
{context}
</context>

Question: {input}
"""
)


user_Query=st.text_input("Enter your query about Field instrument")

import time

if user_Query:
    with st.spinner():
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=vectors.as_retriever(search_kwargs={"k": 4})
        retrieval_chain=create_retrieval_chain(retriever,document_chain)

        start=time.process_time()
        response=retrieval_chain.invoke({'input':user_Query})
        print(f"Response time :{time.process_time()-start}")

        answer = response['answer']
        # Fix escaped backslashes from LLM output
        answer = answer.replace("\\\\", "\\")

        st.success("Answer:")
        st.markdown(answer, unsafe_allow_html=False)

    ## With a streamlit expander
        with st.expander("📄 Sources"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.divider()






