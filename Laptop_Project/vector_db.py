
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 


csv_directory = 'CSVs/'


def upload_csv(file):
    with open(csv_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_csv(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    return documents

file_path = "laptop_price - dataset.csv"
documents = load_csv(file_path)
# print(len(documents))



# Create chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


text_chunks = create_chunks(documents)
# print("Chunks count: ", len(text_chunks))




# Setting up embedding model Huggingface embedding
model_name = "BAAI/bge-small-en"

def get_embedding_model(model_name):
    embeddings = HuggingFaceEmbeddings(model= model_name)
    return embeddings



# FAISS

FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(model_name))
faiss_db.save_local(FAISS_DB_PATH)