# 💬 RAG App with CSV Upload, Chat History & Currency Conversion

This application is a **Retrieval-Augmented Generation (RAG)** system built with **Streamlit** and **LangChain**, enabling users to:

- Upload a CSV file.
- Ask questions based on the CSV data.
- Get answers powered by an LLM (LLaMA 3 via Groq).
- Maintain chat history for contextual understanding.
- Automatically **convert currency values** in **€ (Euro)** or **$ (Dollar)** to **₹ (INR)** in the responses.

---

## 🔧 Features

✅ Upload any CSV file  
✅ Ask natural language questions about the data  
✅ Get contextual answers using LLaMA 3  
✅ Chat history for better multi-turn conversations  
✅ Converts currency values (€, $) to ₹ in answers  
✅ Uses **FAISS** for efficient vector search  
✅ Built-in with **HuggingFace embeddings**

---

## 🛠️ Tech Stack

- **Streamlit** – Web interface
- **LangChain** – LLM & Retrieval management
- **Groq API** – For calling LLaMA 3 (70B)
- **HuggingFace Embeddings** – Document vectorization
- **FAISS** – Vector search
- **Regex + Prompt Engineering** – Currency conversion

---

## 📁 File Structure

