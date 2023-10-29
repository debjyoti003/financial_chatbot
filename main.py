import pickle
import streamlit as st
import os, io
from config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras.add_vertical_space import add_vertical_space

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(
    page_title="Document Chatbot",
    page_icon=":robot_face",
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

st.title("Ask questions from your PDFs 💬")

pdf_docs = st.file_uploader("Upload your PDF to ask question", type = 'pdf', accept_multiple_files= True)

openai_llm = ChatOpenAI(streaming = True, temperature = 0, model_name = 'gpt-3.5-turbo')

if pdf_docs is not None and len(pdf_docs) > 0:
    text = get_pdf_text(pdf_docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text = text)
    # print(f"THis is the chunk: {chunks}")
    get_names = [i.name[:-4] for i in pdf_docs]
    store_name = "__".join(get_names)

    if os.path.exists(f'{store_name}.pkl'):
        with open(f'{store_name}.pkl', 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding = embeddings)
        with open(f'{store_name}.pkl', 'wb') as f:
            pickle.dump(vectorstore, f)


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Please ask your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for message in st.session_state.messages:
                docs = vectorstore.similarity_search(query = message['content'], k =3)
                # print(f"This is the docs: {docs}")
                try:
                    # llm = ChatOpenAI(streaming = True, temperature = 0, model_name = 'gpt-3.5-turbo')
                    chain = load_qa_chain(llm = openai_llm, chain_type = "stuff")
                    response = chain.run(input_documents = docs, question = message['content'])
                except:
                    response = st.error("OpenAI model currently is overloaded with request, please try to ask your question again")
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})