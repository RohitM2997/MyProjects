# from langchain.llms import OpenAI
# api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = api_key  # Needed for LangChain's OpenAI wrapper
# from dotenv import load_dotenv
# load_dotenv()
# import os
# from langchain_ollama import OllamaLLM
from vector_db import faiss_db
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv                                      # to use API keys effectively
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model="llama-3.3-70b-versatile")


# retrive docs
def retrive_docs(query):
    return faiss_db.similarity_search(query)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


# Answer question
custome_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context. Note: 1 Euro = 96.55 rupees, 1 dollar = 86 rupees
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custome_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


