import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.agent.rag_chain import answer_question


def main():
    load_dotenv()

    print("=== AI Customer Support Agent ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        result = answer_question(question)

        print("\nAgent:", result["answer"])

        # Optional: print sources
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. {source['metadata']}")
        
        print()  # empty line for readability


if __name__ == "__main__":
    main()