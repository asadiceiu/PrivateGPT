from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# command to install all the dependencies from requirements.txt
# pip install -r requirements.txt

def pdfGPT():
    load_dotenv()
    st.set_page_config(page_title="PDF GPT", page_icon="🧊", layout="centered", initial_sidebar_state="expanded")
    st.header("PDF GPT: Ask your PDF \U0001F4AC !")
    
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        file_details = {"FileName":pdf.name,"FileType":pdf.type,"FileSize":pdf.size}
        pdf_reader = PdfReader(pdf)
        text = ""
        with st.spinner("Extracting text from PDF..."):
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # split into chunks
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks = splitter.split_text(text)
        st.write(chunks)

        # convert chunks into embeddings
        with st.spinner("Converting text to embeddings..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
        user_question = st.text_input("Ask your question here")
        if user_question:
            with st.spinner("Searching for answer..."):
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(response)
                    st.write(cb)
            st.write(response)


def chatGPT():
    load_dotenv()
    st.set_page_config(page_title="Chat GPT", page_icon="🧊", layout="centered", initial_sidebar_state="expanded")
    st.header("Chat GPT: Chat with your OpenAI!")
    user_question = st.text_input("Ask your question here")
    if user_question:
        with st.spinner("Searching for answer..."):
            llm = OpenAI(temperature=0.9)
            with get_openai_callback() as cb:
                response = llm(user_question)
                st.write(response)
                st.write(cb)

# start streamlit app with command: streamlit run app.py


    

if __name__ == "__main__":
    chatGPT()