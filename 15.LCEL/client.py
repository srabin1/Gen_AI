import requests
import streamlit as st

def get_groq_response(input_text):
    payload = {
        "input": {
            "language": "French",
            "text": input_text
        },
        "config": {},
        "kwargs": {}
    }

    response = requests.post(
        "http://127.0.0.1:8000/chain/invoke",
        json=payload  # ✅ important
    )

    # Helpful error message if something goes wrong
    response.raise_for_status()

    return response.json()

st.title("LLM Application Using LCEL")
input_text = st.text_input("Enter the text you want to convert to french")

if input_text:
    result = get_groq_response(input_text)
    st.write(result)               # full response
    st.success(result["output"])   # just the translated text
