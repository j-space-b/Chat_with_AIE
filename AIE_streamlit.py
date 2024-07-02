import streamlit as st
import os
import urllib.request
import io
from PyPDF2 import PdfReader
import openai
import faiss
import numpy as np

# [Keep the existing imports and page configuration]

# Move compute_embeddings outside of load_document
def compute_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

@st.cache_data
def load_document():
    file_id = "18qcIHc8lGJiKztyRKd5m7n2q0b1jvS-v"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
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
        
        # Compute embeddings for each chunk of text
        embeddings = compute_embeddings(texts)
        
        # Create FAISS index and add embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        return index, texts
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None, None

if openai_api_key:
    index, texts = load_document()

    if index is not None and texts is not None:
        # [Keep the existing chat interface code]

        if prompt := st.chat_input("Ask about the 2024 AIE World Summit"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.container():
                st.markdown(f"<div class='chat-message user'>" +
                            f"<div class='message'>{prompt}</div></div>",
                            unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                try:
                    # Compute the embedding for the query
                    query_embedding = compute_embeddings([prompt])
                    D, I = index.search(query_embedding, k=1)  # search for the nearest neighbors
                    
                    # Get the best matching document
                    response = texts[I[0][0]]

                    with st.container():
                        st.markdown(f"<div class='chat-message bot'>" +
                                    f"<div class='message'>{response}</div></div>",
                                    unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

else:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
