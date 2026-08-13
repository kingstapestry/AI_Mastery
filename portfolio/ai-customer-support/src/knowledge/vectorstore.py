from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# ======================
# Function: Create Vectorstore
# ======================
def create_vectorstore(chunks: list, persist_directory: str = "data/processed/chroma"):
    """
    Create a Chroma vector store from document chunks.
    """
    embedding_model = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    return vectorstore

# ======================
# Function: Load Vectorstore
# ======================
def load_vectorstore(persist_directory: str = "data/processed/chroma"):
    """
    Load an existing Chroma vector store from disk.
    """
    embedding_model = OpenAIEmbeddings()

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    return vectorstore

# ======================
# Function: Retrieve Relevant Docs
# ======================
def retrieve_relevant_docs(query: str, k: int = 6):
    """
    Retrieve the most relevant documents for a given query.
    """
    vectorstore = load_vectorstore()

    relevant_docs = vectorstore.similarity_search(query=query, k=k)
    
    return relevant_docs