import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Set page configuration
st.set_page_config(page_title="Chat with 2024 AIE World Summit", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #F0F2F6;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #E6F3FF;
        justify-content: flex-end;
    }
    .chat-message.bot {
        background-color: #F0F0F0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    st.markdown("Don't have an API key? [Get one here](https://platform.openai.com/account/api-keys)")

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    st.markdown("---")
    st.markdown("Made with ❤️ by Jonathan Bennion")
    st.markdown("---")
    st.markdown("Transcripts summarized from both days, PII obfuscated.")

# Main content
st.markdown("<h1 class='main-header'>Chat with 2024 AIE World Summit Talk Summaries</h1>", unsafe_allow_html=True)


@st.cache_resource
def load_document():
    loader = PyPDFLoader("https://drive.google.com/file/d/18qcIHc8lGJiKztyRKd5m7n2q0b1jvS-v/view?usp=sharing")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50) # not going to lose any info lol
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore


if openai_api_key:
    vectorstore = load_document()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"<div class='chat-message {message['role']}'>" +
                        f"<div class='message'>{message['content']}</div></div>",
                        unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about the 2024 AIE World Summit"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.container():
            st.markdown(f"<div class='chat-message user'>" +
                        f"<div class='message'>{prompt}</div></div>",
                        unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0, model_name='gpt-4o'),
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in
                                                                 st.session_state.messages]})
            response = result['answer']

        with st.container():
            st.markdown(f"<div class='chat-message bot'>" +
                        f"<div class='message'>{response}</div></div>",
                        unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
