from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.5,
) # type: ignore

model = ChatHuggingFace(llm=llm)


st.header("Research Tool")
user_input =  st.text_input("Enter your prompt")

if st.button('Summarise'):
    response = model.invoke(user_input)
    st.write(response.content)    # st.write() is a flexible output function that can display almost anything on a Streamlit app page.