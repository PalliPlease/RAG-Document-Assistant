
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/me-cse-brochure.pdf"
PERSIST_DIR = "vectordb"

def load_doc():
    loader = PyPDFLoader(DATA_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")
    return docs

def split_doc(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
        separators = ["\n\n", "\n", ".", "?", "!", " "]
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = PERSIST_DIR
    )

    vectordb.persist()
    print("VectorDB created")
    return vectordb

if __name__ == "__main__":
    docs = load_doc()
    chunks = split_doc(docs)
    create_vector(chunks)
    print("Ingestion complete")


#pulls in cmd terminal

#ollama pull llama3
#ollama pull nomic-embed-text   # for embeddings
