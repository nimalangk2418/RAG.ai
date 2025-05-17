import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# API configuration 
genai.configure(api_key=os.getenv("Google_api"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Cache the HF Embeddings to avoid slow reload of the Embeddings
@st.cache_resource(show_spinner="Loading Embedding Model ...")
# model
def embeddings():
    #return (HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2"))
    return(HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"))

embedding_model = embeddings()

# USER Interface
# st.markdown("<body style='background-color:red;'>", unsafe_allow_html=True)
st.header("RAG Assitant : :red[HF Embeddings + Gemini LLM]")
st.subheader(":orange[Your AI Doc Assistant] 🤖")
upload_file = st.file_uploader(label = "Upload the PDF Reader",
                               type = ["pdf"])                         

if upload_file:
    raw_text = ""
    pdf = PdfReader(upload_file)
    for i,page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text+=text
    if raw_text.strip():
        document = Document(page_content=raw_text)
        # Using CharaterTextSplitter create chunks and pass into the model
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap =200 )
        chunks = splitter.split_documents([document])
        
        # store the chunks into FAISS vectordb
        chunk_pieces = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(chunk_pieces,embedding_model)
        retriever = vector_db.as_retriever() # Retrieve the vectors
        
        st.success("Embedding are Generated ✅. Ask your Questions 📝")
        user_input = st.text_input(label = "Enter your Questions ❓")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analyzing the document ......"):
                relavant_docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in relavant_docs])
                prompt = f'''You are an expert assistant. 
                Use the context to answer the Query.
                If unsure or Information not available in the doc,
                pass the message - "💥 Information is not available. Look into Other Sources 🔍 "
                context = {context}
                query = {user_input}
                Answer:'''
                response = gemini_model.generate_content(prompt)
                st.markdown("Answer:")
                st.write(response.text)
    else:
        st.warning(''' 	
⚠️ No Text could be Extracted from PDF. Please Upload as readable PDF 📃''')
        