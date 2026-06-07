from datetime import datetime


"""
Personal Finance Manager

A single-file command-line application for managing personal finances.

Features:
    * Add income and expense transactions
    * View and search transaction history
    * Generate financial summaries
    * Save and load transaction data
    * Delete transactions

Technology:
    * Python Standard Library only

Learning Objectives:
    * Object-Oriented Programming (OOP)
    * File Handling
    * Data Structures
    * Error Handling
    * Program Organization
    * User Input Validation

This project is designed as a comprehensive Python fundamentals exercise
and serves as a practical demonstration of core programming concepts.
"""


class Transaction:
    def __init__(self, amount, category, description, tx_type, transaction_date):
        self.amount = amount
        self.category = category
        self.description = description
        self.tx_type = tx_type
        self.date = transaction_date

    def __str__(self):
        return (
            f"[{self.date}] "
            f"{self.category} | "
            f"{self.description} | "
            f"{self.tx_type} | "
            f"${self.amount}"
        )


transactions = []


def add_transaction():
    print("\n--- Add Transaction ---")

    try:
        amount = float(input("Amount (+/-): "))
    except ValueError:
        print("Please enter a valid number.")
        return

    category = input("Category: ")
    description = input("Description: ")
    tx_type = input("Transaction Type (income/expense): ")

    new_transaction = Transaction(
        amount,
        category,
        description,
        tx_type,
        datetime.now().strftime("%Y-%m-%d")
    )

    transactions.append(new_transaction)


def view_transactions():
    print("\n--- Transactions ---")

    if not transactions:
        print("No transactions found.")
        return
    
    for i, transaction in enumerate(transactions, start=1):
        print(f"{i}. {transaction}")


def search_transactions():
    print("\n--- Search Transactions ---")

    search_term = input("Enter search term (eg. food): ").lower()

    if not search_term:
        print("Please enter a search term.")
        return

    matches = [
        transaction
        for transaction in transactions
        if search_term in transaction.category.lower()
        or search_term in transaction.description.lower()
    ]

    if not matches:
        print("Transaction not found!")
        return
    
    print(f"\nFound {len(matches)} matching transaction(s):")

    for i, transaction in enumerate(matches, start=1):
        print(f"{i}. {transaction}")


def show_statistics():
    print("\n--- Statistics ---")

    if not transactions:
        print("No transaction statistics available.")
        return

    total_income = sum(transaction.amount for transaction in transactions if transaction.tx_type == "income")
    total_expenses = sum(transaction.amount for transaction in transactions if transaction.tx_type == "expense")
    largest_income = max([t for t in transactions if t.tx_type == "income"], key=lambda t: t.amount, default=None)
    largest_expense = max([t for t in transactions if t.tx_type == "expense"], key=lambda t: t.amount, default=None)
    average_transaction = sum(transaction.amount for transaction in transactions) / len(transactions)
    net_balance = sum(t.amount for t in transactions)

    print(f"Total Income: {total_income}")
    print(f"Total Expenses: {total_expenses}")
    print(f"Largest Income: {largest_income}")
    print(f"Largest Expense: {largest_expense}")
    print(f"Average Transaction: {average_transaction}")
    print(f"\nNet Balance: {net_balance}")

    print("\n--- By Category ---")

    category_totals = {}

    for transaction in transactions:
        category = transaction.category

        if category not in category_totals:
            category_totals[category] = transaction.amount
        else:
            category_totals[category] += transaction.amount

    for key, value in category_totals.items():
        print(f"{key}: {value}")


def save_data():
    file_path = "side_projects/finance_manager/statements.txt"
    with open(file_path, "w") as file:
        for transaction in transactions:
            file.write(
                f"{transaction.amount},"
                f"{transaction.category},"
                f"{transaction.description},"
                f"{transaction.tx_type},"
                f"{transaction.date}\n"
            )

    print(f"\n--- Saved Transactions to {file_path} ---")


def load_data():
    file_path = "side_projects/finance_manager/statements.txt"

    try:
        with open(file_path, "r") as file:
            transactions.clear()
            for line in file:
                line = line.strip()
                if not line:
                    continue        # skip empty lines

                parts = [part.strip() for part in line.split(",")]

                amount, category, description, tx_type, transaction_date = parts

                amount = float(amount)

                transactions.append(
                    Transaction(
                        amount, 
                        category, 
                        description, 
                        tx_type, 
                        transaction_date
                    )
                )

        print(f"\n--- Loaded Transactions from {file_path} ---")
    except FileNotFoundError:
        print(f"File not found: {file_path}")


def delete_transaction():
    print("\n--- Delete Transactions ---")

    if not transactions:
        print("No transactions found.")
        return
    
    view_transactions()

    try:
        choice = int(input("\nTransaction number to delete: "))
    except ValueError:
        print("Please enter a valid number.")
        return
    
    if choice < 1 or choice > len(transactions):
        print("Invalid transaction number")
        return

    print("Delete this transaction?")
    print(f"{transactions[choice - 1]}")
    
    yes_no = input("\nY/N: ").strip().lower()

    if yes_no == "y":
        deleted_transaction = transactions.pop(choice - 1)
        print(f"Deleted: {deleted_transaction}")
    else:
        print("Operation cancelled.")
        return


def display_menu():
    print("\n===== PERSONAL FINANCE MANAGER =====")
    print("1. Add Transaction")
    print("2. View Transactions")
    print("3. Search Transactions")
    print("4. Show Statistics")
    print("5. Save Data")
    print("6. Load Data")
    print("7. Delete Transaction")
    print("8. Exit")


def main():
    while True:
        display_menu()

        choice = input("\nChoose an option: ")

        if choice == "1":
            add_transaction()

        elif choice == "2":
            view_transactions()

        elif choice == "3":
            search_transactions()

        elif choice == "4":
            show_statistics()

        elif choice == "5":
            save_data()

        elif choice == "6":
            load_data()

        elif choice == "7":
            delete_transaction()

        elif choice == "8":
            print("Goodbye!")
            break

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()