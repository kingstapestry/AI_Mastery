import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.knowledge.vectorstore import retrieve_relevant_docs


def main():
    load_dotenv()

    query = "Give me information on the caution and warnings of this Bosch appliance."
    docs = retrieve_relevant_docs(query)

    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---\n")
        print(doc.page_content)


if __name__ == "__main__":
    main()