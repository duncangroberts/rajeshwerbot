import streamlit as st
import pandas as pd
from llama_cpp import Llama
import os
import textwrap
import io
import sys

# Function to clean and dedent code
def clean_and_dedent_code(code):
    # Split into lines, remove empty lines, and strip leading/trailing whitespace from each line
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    # Join back and then dedent
    return textwrap.dedent('\n'.join(lines))

# --- App Configuration ---
st.set_page_config(page_title="Rajeshwerbot", layout="wide")
st.title("Rajeshwerbot")

# --- Model Loading ---
if 'llm' not in st.session_state:
    st.session_state.llm = None

if st.session_state.llm is None:
    with st.spinner("Rajeshwerbot is waking up... This might take a moment."):
        model_path = os.path.join("model", "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf")
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please make sure the model is in the 'model' directory.")
            st.stop()
        try:
            st.session_state.llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            st.success("Rajeshwerbot is ready!")
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

llm = st.session_state.llm

# --- Main App Logic ---
st.write("Welcome to Rajeshwerbot. Please upload a CSV file to begin.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data
    st.write("Data loaded successfully:")
    st.dataframe(data)

# --- Chat Interface ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data' in st.session_state:
    st.sidebar.title("Ask a question")
    user_question = st.sidebar.text_input("What would you like to know about the data?", key="user_question")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Thinking..."):
            # --- LLM Interaction ---
            prompt = f"""You are a data analyst. The user has loaded a dataset with the following columns: {', '.join(st.session_state.data.columns)}. The data is available in a pandas DataFrame named 'df'.
            The user's question is: '{user_question}'.
            
            Please provide a concise natural language answer to the user's question based on the data. If you need to perform calculations or data analysis to answer, assume you have access to the 'df' DataFrame and can perform operations on it to derive the answer. Do not provide any code, just the natural language answer.
            """

            response = llm(prompt, max_tokens=500, echo=False)
            generated_text = response['choices'][0]['text'].strip()
            
            st.session_state.messages.append({"role": "assistant", "content": generated_text})

    # --- Display Conversation ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Model Loading (GPU enabled) ---
if 'llm' not in st.session_state:
    st.session_state.llm = None

if st.session_state.llm is None:
    with st.spinner("Rajeshwerbot is waking up... This might take a moment."):
        model_path = os.path.join("model", "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf")
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please make sure the model is in the 'model' directory.")
            st.stop()
        try:
            # n_gpu_layers=-1 attempts to offload all layers to the GPU
            st.session_state.llm = Llama(model_path=model_path, n_ctx=2048, verbose=True, n_gpu_layers=999)
            st.success("Rajeshwerbot is ready!")
            if st.session_state.llm.n_gpu_layers > 0:
                st.info(f"Successfully offloaded {st.session_state.llm.n_gpu_layers} layers to GPU.")
            else:
                st.warning("No layers offloaded to GPU. Running on CPU.")
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()
