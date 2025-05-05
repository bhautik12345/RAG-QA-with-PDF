import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from PyPDF2 import PdfReader
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
import nltk
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF
from langchain.schema import Document

load_dotenv()

os.environ['HF_Token'] = os.getenv('HF_Token')
# os.environ['GOOGLE_API_KEY']=os.getenv('GEMINI_API_KEY')

# 🛠️ Set Streamlit page configuration
st.set_page_config(
    page_title="Conversational RAG",
    page_icon="📄💬",
    layout="centered",  # Options: "centered" or "wide"
    initial_sidebar_state="auto"
)

# 🌟 App title and welcome message
st.title("📄💬 Conversational RAG with PDF Uploads & Chat History")
st.markdown(
    """
    Welcome to **Conversational RAG** – your intelligent assistant for interacting with PDFs!  
    🗂️ **Upload your PDF files** and  
    💬 **Chat directly with their content**,  
    all while keeping track of your conversation history.
    """
)
st.sidebar.info(
    "📎 **Note:**\n"
    "- You can upload **multiple PDF files** at once.\n"
    "- For best performance, keep each file under **10 MB**.\n"
    "- ⛔ Some PDFs may have content that cannot be extracted (e.g., scanned images or protected documents). "
    "We apologize if this happens."
)


google_api_key = st.text_input('Google AI Studio API',type='password')
os.environ['GOOGLE_API_KEY'] = google_api_key

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


if google_api_key:
    model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

    #chat interface

    session_id = st.text_input('Session ID',value='Default Session')

    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader('Choose A PDf From Browse',type='pdf',accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if uploaded_file.size > 10 * 1024 * 1024: 
            st.error("🚫 File size exceeds 10 MB. Please upload a smaller PDF.")
            st.stop()

    if uploaded_files:
        # documents = []
        # temppdf = f'./temp.pdf'
        # with open(temppdf,'wb') as file:
        #     file.write(uploaded_file.read())
        #     file_name = uploaded_file.name

        # loader = PyPDFLoader(temppdf)

        # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        #     tmp.write(uploaded_file.read())
        #     tmp_path = tmp.name

        # loader = PyMuPDFLoader(tmp_path)
    
        # docs = loader.load()
        # documents.extend(docs)
        
        #split documents and store in database
        # splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=500)
        # final_doc = splitter.split_documents(documents)
        # db = FAISS.from_documents(documents=final_doc,embedding=embedding)
        # retriever = db.as_retriever()

        text = get_pdf_text(uploaded_files)
        chunks = get_text_chunks(text)
        vectorstore = get_vectorstore(chunks)
        retriever = vectorstore.as_retriever()

        #maintain chat history with the help of history aware retriever
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ('system',contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])
        history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If the answer is not known, respond by saying you don't know. "
            "However, if you do know the answer but it's not included in the provided PDF content, "
            "first inform the user that you can provide the answer.  "
            " If the user requests a detailed summary, provide a longer response; "
            "if they prefer something concise, keep it short."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])
        document_chain = create_stuff_documents_chain(llm=model,prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,document_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key='input',history_messages_key='chat_history',output_messages_key='answer')


        user_prompt = st.text_input('Write your query related to PDf')

        if user_prompt:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({'input':user_prompt},config={'configurable':{'session_id':session_id}})
            
            # st.write(st.session_state.store)
            st.write('Assistant :')
            st.success(response['answer'])
            st.write('chat history ',session_history.messages)



