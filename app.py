from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe

model = LocalLLM(
    api_base = "http://localhost:11434/v1",
    model = "llama3:8b"
)

st.title("Data Analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write(data.head(5))
    
    df = SmartDataframe(data, config={"llm": model})
    prompt = st.text_area("Enter your prompt: ")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))