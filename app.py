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
            # n_gpu_layers=-1 attempts to offload all layers to the GPU
            st.session_state.llm = Llama(model_path=model_path, n_ctx=2048, verbose=True, n_gpu_layers=999)
            st.success("Rajeshwerbot is ready!")
            st.info("Attempting to offload layers to GPU. Check your console for detailed GPU usage information.")
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

    # Generate schema for the LLM
    schema_info = ""
    for col in data.columns:
        dtype = data[col].dtype
        schema_info += f"- Column '{col}' (Type: {dtype})\n"
        if dtype == 'object': # Likely a string/categorical column
            unique_values = data[col].unique()
            # Limit unique values to avoid overwhelming the prompt
            if len(unique_values) < 20:
                schema_info += f"  Unique values: {list(unique_values)}\n"
            else:
                schema_info += f"  (Too many unique values to list)\n"
    st.session_state.schema_info = schema_info

# --- Chat Interface ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data' in st.session_state:
    st.sidebar.title("Ask a question")
    user_question = st.sidebar.text_input("What would you like to know about the data?", key="user_question")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.spinner("Rajeshwerbot is contemplating the multipolarity of currencies while generating your data query..."):
            # --- Step 1: LLM generates Python code for data analysis ---
            code_prompt = f"""You are an intelligent data analyst. The user has loaded a dataset with the following schema:
{st.session_state.schema_info}

The data is available in a pandas DataFrame named 'df'.
            The user's question is: '{user_question}'.
            
            Your task is to generate a Python script using pandas to perform the necessary data analysis to answer the user's question. The script should print the final, concise answer to standard output. Do not include any comments, extra text, or code to load the data. Ensure the script is complete and executable.
            
            **Important Guidelines for Code Generation:**
            1.  **Column Inference:** You MUST infer the most relevant column(s) from the user's question by comparing keywords in the question to the provided schema. For example, if the user asks about '2026 automation' and the schema has 'Automation_Score_2026', you should use 'Automation_Score_2026'. Always use the exact column names from the provided schema.
            2.  **Value Inference:** Infer the target values or conditions. For text, use `.astype(str).str.contains('keyword', case=False, na=False)` for robust matching.
            3.  **Direct Output:** The script must `print()` ONLY the final answer. If no data is found or an answer cannot be computed, print "NO_DATA_FOUND".
            
            Here are some examples:
            
            Example 1: Count tasks fully automatable in 2026 (dynamic column inference)
            ```python
            # User question: "How many tasks are fully automatable in 2026?"
            # Inferred column (based on schema): 'Automation_Score_2026' (if 'Automation_Potential_2026' is not present)
            count = df[df['Automation_Score_2026'].astype(str).str.contains('Fully Automatable', case=False, na=False)].shape[0]
            print(count)
            ```
            
            Example 2: Average of a numerical column
            ```python
            # User question: "What is the average value of X?"
            # Inferred column: 'X_Column'
            average_value = df['Numerical_Column'].mean()
            print(average_value)
            ```
            
            Example 3: Filtered data (print specific column)
            ```python
            # User question: "Show me data for categories containing 'Specific'."
            # Inferred column: 'Category'
            filtered_data = df[df['Category'].astype(str).str.contains('Specific', case=False, na=False)]
            if not filtered_data.empty:
                print(filtered_data['Relevant_Column'].to_string(index=False))
            else:
                print("NO_DATA_FOUND")
            ```
            
            Now, generate the script for the user's question.
            ```python
            """

            code_response = llm(code_prompt, max_tokens=500, stop=["```"], echo=False)
            generated_code_text = code_response['choices'][0]['text'].strip()
            
            # Extract code between ```python and ```
            if '```python' in generated_code_text:
                code_to_execute = generated_code_text.split('```python')[1].split('```')[0].strip()
            else:
                code_to_execute = generated_code_text

            # Clean and dedent the code before execution
            code_to_execute = clean_and_dedent_code(code_to_execute)

        with st.spinner("Executing complex financial algorithms, or maybe just counting rows. Either way, it's about the data..."):
            # --- Step 2: Execute the generated code and capture output ---
            captured_output = io.StringIO()
            sys.stdout = captured_output
            try:
                exec_namespace = {'df': st.session_state.data.copy(), 'pd': pd}
                exec(code_to_execute, exec_namespace)
                analysis_result = captured_output.getvalue().strip()
            except KeyError as ke:
                analysis_result = f"Error: Column not found. It seems the column '{ke}' does not exist in the dataset. Please check the column name or rephrase your question."
            except Exception as e:
                analysis_result = f"Error during data analysis: {e}"
            finally:
                sys.stdout = sys.__stdout__ # Restore stdout

        with st.spinner("Synthesizing insights, like predicting the next global reserve currency..."):
            # --- Step 3: LLM generates natural language answer ---
            nl_prompt = f"""You are a data analyst. The user asked: '{user_question}'.
            Based on your analysis, the following result was obtained: '{analysis_result}'.
            
            Provide a concise, natural language answer to the user's question, incorporating the analysis result. Be direct and avoid conversational filler.
            
            If the analysis result is "NO_DATA_FOUND", state clearly that the information could not be found in the data. If the analysis result indicates an error, explain that the data could not be analyzed as requested.
            """
            nl_response = llm(nl_prompt, max_tokens=500, echo=False)
            final_answer = nl_response['choices'][0]['text'].strip()
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # --- Display Conversation ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])