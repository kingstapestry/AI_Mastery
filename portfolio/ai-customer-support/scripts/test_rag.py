import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.agent.rag_chain import answer_question


def main():
    load_dotenv()

    question = "What are the steps to install the cooktop?"
    result = answer_question(question)

    print("\nQuestion:", question)
    print("\nAnswer:", result["answer"])

    print("\nSources:")
    for i, source in enumerate(result["sources"]):
        print(f"\n--- Source {i+1} ---")
        print("Preview:", source["content_preview"])
        print("Metadata:", source["metadata"])


if __name__ == "__main__":
    main()