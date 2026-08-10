import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.knowledge.ingestion import load_documents, split_documents
from src.knowledge.vectorstore import create_vectorstore


def main():
    load_dotenv()

    print("Loading documents...")
    documents = load_documents("data/raw")

    print("Splitting elements...")
    chunks = split_documents(documents)

    print("Creating vectore store...")
    vectorstore = create_vectorstore(chunks)

    print("Knowledge base created successfully!")
    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")


if __name__ == "__main__":
    main()