import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================
# Function: Load Documents
# ======================
def load_documents(folder_path: str) -> list:
    """
    Load all PDF files from a given folder and return them as LangChain documents.
    """
    documents = []

    files = os.listdir(folder_path)                             # get all files in the folder path

    for filename in files:                                      # loop through the files
        if filename.endswith(".pdf"):           
            full_path = os.path.join(folder_path, filename)     # if file ends with ".pdf", create full path

            loader = PyPDFLoader(full_path)                     # create instance
            loaded_docs = loader.load()                         # call load on that instance
            documents.extend(loaded_docs)                       # add loaded documents to list

    return documents

# ======================
# Function: Split Documents
# ======================
def split_documents(documents: list) -> list:
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)           # use the splitter's method to split documents
    return chunks                                               # return the result directly