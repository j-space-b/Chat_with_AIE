import streamlit as st
import os
import urllib.request
import io
from PyPDF2 import PdfReader
import openai
import faiss
import numpy as np

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
        openai.api_key = openai_api_key

    st.markdown("---")
    st.markdown("Made with ❤️ by Jonathan Bennion")
    st.markdown("---")
    st.markdown("Transcripts summarized from both days, PII obfuscated.")

# Main content
st.markdown("<h1 class='main-header'>Chat with 2024 AIE World Summit Talk Summaries</h1>", unsafe_allow_html=True)

@st.cache_data
def load_document():
    file_id = "18qcIHc8lGJiKztyRKd5m7n2q0b1jvS-v"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    with urllib.request.urlopen(download_url) as response:
        pdf_content = response.read()
    
    pdf_file = io.BytesIO(pdf_content)
    
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    chunk_size = 100
    texts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Function to compute embeddings using OpenAI
    def compute_embeddings(texts):
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            embeddings.append(response['data'][0]['embedding'])
        return np.array(embeddings)
    
    # Compute embeddings for each chunk of text
    embeddings = compute_embeddings(texts)
    
    # Create FAISS index and add embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, texts

if openai_api_key:
    vectorstore, texts = load_document()

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
            # Compute the embedding for the query
            query_embedding = compute_embeddings([prompt])
            D, I = vectorstore.search(query_embedding, k=1)  # search for the nearest neighbors
            
            # Get the best matching document
            response = texts[I[0][0]]

        with st.container():
            st.markdown(f"<div class='chat-message bot'>" +
                        f"<div class='message'>{response}</div></div>",
                        unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
