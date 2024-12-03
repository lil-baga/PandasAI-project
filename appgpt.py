from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

llm = OpenAI(api_token="redacted")

st.title("Data Analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        try:
            data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            data = pd.read_csv(uploaded_file, encoding='utf-8')

        st.write("### Preview of Uploaded Data:")
        st.write(data.head(5))

        df = SmartDataframe(data, config={"llm": llm})

        st.write("### Example Prompts:")
        st.write("- What is the average value in column X?")
        st.write("- Show me the top 5 rows sorted by column Y.")
        st.write("- How many unique values are in column Z?")

        prompt = st.text_area("Enter your prompt:")
        output_type = st.selectbox("Select the expected output type:", ["Text", "Table", "Image"])

        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    try:
                        response = df.chat(prompt)

                        if output_type == "Text":
                            st.write(response)
                        elif output_type == "Table":
                            try:
                                response_df = pd.DataFrame(response)
                                st.write(response_df)
                            except Exception:
                                st.error("Could not interpret the response as a table. Showing as text instead:")
                                st.write(response)
                        elif output_type == "Image":
                            try:
                                st.write(response)
                                st.image("exports/charts/temp_chart.png")
                            except Exception as e:
                                st.error("Could not interpret the response as a image. Showing as text instead:")
                                st.write(response)
                    except Exception as e:
                        st.error(f"Error processing the prompt: {e}")
            else:
                st.warning("Please enter a prompt to generate a response.")

    except Exception as e:
        st.error(f"Error reading the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")

