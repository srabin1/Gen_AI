import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OPENAI"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

# temperature = [0,1], 0 model is not creative and will
# produce the same response asking the same question, 1 model gets creative

def generate_response(question, api_key, model_name, temperature, max_tokens):
    llm_client = ChatOpenAI(
        model=model_name,
        api_key=api_key,          
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt | llm_client | output_parser
    return chain.invoke({"question": question})

# Title
st.title("Enhanced Q&A Chatbot With OpenAI")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

model_name = st.sidebar.selectbox("Select OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar")
else:
    st.write("Please provide the user input")
