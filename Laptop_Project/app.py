import streamlit as st
from rag_pipe import answer_query, retrive_docs
from rag_pipe import llm


st.title("Laptop Finder")
user_query = st.text_area("Enter your prompt: ", height=150, placeholder = "Ask anything!")
ask_question = st.button("Ask about laptop")


if ask_question:
    if user_query:
        st.chat_message("user").write(user_query)

        #RAG pipeline
        retrived_docs = retrive_docs(user_query)
        response = answer_query(documents=retrived_docs, model=llm, query=user_query)
        #fixed_response ="Hi this is fixed response"
        st.chat_message("laptop finder").write(response.content)
    
    else:
        st.error("kindly write a question")
