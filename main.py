import streamlit as st
from langchain_community.llms import OpenAI
from langchain_helper import get_qa_chain, create_vector_db

st.title("Codebasics Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

API_KEY = st.text_input("OpenAI API Key", type="password")
# OPENAI_API_KEY=openai_api_key
if not API_KEY:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

# Create OpenAI model
llm = OpenAI(openai_api_key=API_KEY, temperature=0.2)

question = st.text_input("Question: ")


if question:
    chain = get_qa_chain(llm)
    response = chain.invoke(question)
    # response = chain.invoke({"input": question})

    st.header("Answer")
    st.write(response)