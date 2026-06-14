from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import json
from datetime import datetime
import time

"""
LESSON 25 - ADVANCED: Professional OOP Patterns
Concepts Explored: 
  - Dataclass Inheritance & Field Resolution
  - Abstract Base Classes (ABCs) as Architectural Contracts
  - Encapsulation via Protected Attributes
  - @property for Computed, Read-Only States
  - @staticmethod vs @classmethod Utilities
  - Context Managers for Resource Control
"""

# ==================== 1. BASE DATA CLASS ====================
@dataclass
class Person:
    """
    Base class for any person using Python's @dataclass decorator.
    
    Why @dataclass? It automatically generates __init__(), __repr__(), and 
    structural comparisons, reducing boilerplate code significantly.
    """
    name: str
    email: str


# ==================== 2. INHERITANCE ====================
@dataclass
class Customer(Person): # Inherits from Person
    """
    Customer class extending Person.
    
    CRITICAL DATACLASS RULE: Python builds the internal constructor by placing 
    parent class fields FIRST, followed by child class fields. 
    Therefore, the true argument order expected here is:
    (name, email, customer_id, plan, monthly_fee, join_date, is_active)
    """
    customer_id: int
    plan: str
    monthly_fee: float
    join_date: str
    is_active: bool = True # Default fields must always come last in dataclasses

    def __post_init__(self):
        """
        The __post_init__ method is automatically called by dataclasses after 
        the standard __init__ assignment completes.
        
        Why use it? It is the ideal architectural hook for data validation 
        and state normalization when using dataclasses.
        """
        if self.monthly_fee <= 0:
            raise ValueError("Monthly fee must be positive")


# ==================== 3. ABSTRACT BASE CLASS ====================
class BaseManager(ABC):
    """
    Abstract Base Class (ABC) serving as an interface contract.
    
    Why use ABCs? 
    1. It cannot be instantiated directly (e.g., BaseManager() throws an error).
    2. It guarantees that any concrete subclass MUST implement every method 
       marked with @abstractmethod, ensuring reliable interface conformity.
    """
    
    @abstractmethod
    def add(self, item):
        """Forces child managers to implement a method to add records."""
        pass
    
    @abstractmethod
    def get_all(self):
        """Forces child managers to implement a method to retrieve all records."""
        pass
    
    @abstractmethod
    def save(self):
        """Forces child managers to implement data persistence."""
        pass
    
    @abstractmethod
    def load(self):
        """Forces child managers to implement data retrieval."""
        pass


# ==================== 4. CONCRETE MANAGER WITH ADVANCED FEATURES ====================
class CustomerManager(BaseManager):
    """
    Concrete implementation of BaseManager enforcing structural contracts, 
    encapsulation, and advanced method decorators.
    """
    
    def __init__(self):
        # The single underscore (_) is an OOP convention signaling a 'Protected' attribute.
        # It warns other developers not to access or mutate `self._customers` directly 
        # from outside the class, protecting internal data integrity.
        self._customers: List[Customer] = [] 
        self.data_file = "advanced_python\customers.json"
    
    # ==================== PROPERTY (Getter) ====================
    @property
    def customer_count(self) -> int:
        """
        Exposes a dynamic state as an attribute instead of a method.
        
        Why use @property? It acts like a calculated field. Outside this class,
        you access this via `manager.customer_count` (no parentheses!), preventing 
        external code from accidentally modifying it since it has no setter.
        """
        return len(self._customers)
    
    @property
    def active_customers(self) -> List[Customer]:
        """Dynamically filters and returns only active customers on-the-fly."""
        return [c for c in self._customers if c.is_active]
    
    # ==================== STATIC METHOD ====================
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """
        A completely isolated utility function nested inside the class namespace.
        
        Why @staticmethod? It doesn't look at, modify, or require access to instance 
        states (`self`) or class states (`cls`). It lives here purely because email 
        validation is conceptually tied to customer management.
        """
        return "@" in email and "." in email
    
    # ==================== CLASS METHOD ====================
    @classmethod
    def create_from_dict(cls, data: list) -> 'CustomerManager':
        """
        An alternative constructor factory method.
        
        Why @classmethod? Instead of operating on an existing instance (`self`), 
        it takes the class itself (`cls`) as an argument. This allows you to set up
        and return a pre-configured instance of `CustomerManager` directly from raw data.
        """
        manager = cls() # Dynamically instantiates an instance of this exact class
        for item in data:
            # The double asterisk (**) unpacks dictionary keys/values directly into 
            # the corresponding constructor argument fields.
            customer = Customer(**item) 
            manager.add(customer)
        return manager
    
    def add(self, customer: Customer):
        """Adds a customer after applying strict domain-specific business rules."""
        # Clean architecture: accessing the static utility method within an instance method
        if not self.is_valid_email(customer.email):
            raise ValueError(f"Invalid email: {customer.email}")
        self._customers.append(customer)
        print(f"✅ Added customer: {customer.name}")
    
    def get_all(self) -> List[Customer]:
        """Provides controlled, explicit access to encapsulated data."""
        return self._customers
    
    def save(self):
        """Serializes runtime application objects into permanent JSON data."""
        # vars(c) extracts the underlying dictionary state (__dict__) of the dataclass object
        data = [vars(c) for c in self._customers]
        with open(self.data_file, 'w') as f:
            # default=str handles objects like datetimes or custom classes safely during serialization
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Saved {self.customer_count} customers")
    
    def load(self):
        """Deserializes external JSON records back into live Python Objects."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            # Standard structural reconstructor pattern via dictionary unpacking (**item)
            self._customers = [Customer(**item) for item in data]
            print(f"📂 Loaded {self.customer_count} customers")
        except FileNotFoundError:
            print("📁 No data file found. Starting fresh.")
        except Exception as e:
            print(f"⚠️ Error loading data: {e}. Starting fresh.")


# ==================== 5. CONTEXT MANAGER ====================
class Timer:
    """
    A custom Context Manager designed to seamlessly measure execution duration.
    
    Why use a Context Manager? By defining __enter__ and __exit__, this class 
    can hook into Python's `with` statement block to handle setup and teardown 
    logic automatically, even if errors occur inside the block.
    """
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
    
    def __enter__(self):
        """Triggers exactly when entering the scope of the 'with' block."""
        self.start = time.time()
        print(f"⏱️ Starting: {self.operation_name}")
        return self # Allows you to capture a variable alias using `with Timer(...) as t:`
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Triggers automatically when exiting or escaping the 'with' block.
        
        The arguments exc_type, exc_val, and exc_tb capture exception details 
        if anything crashed inside the block, allowing you to handle or suppress bugs.
        """
        duration = time.time() - self.start
        print(f"✅ Finished: {self.operation_name} | Duration: {duration:.4f} seconds")


# ==================== 6. MAIN DEMO ====================
def main():
    print("=== Advanced OOP Demo ===\n")
    
    # 1. Initialize the concrete orchestrator object
    manager = CustomerManager()
    manager.load()
    
    # 2. Utilize our custom context manager to safely profile execution time
    with Timer("Adding Customers"):
        # Correctly structured parameters following strict Dataclass Field Resolution:
        # Parent fields (name, email) -> Child fields (customer_id, plan, monthly_fee, join_date)
        c1 = Customer("Michael King", "michael@example.com", 1, "Premium", 29.99, "2026-06-01")
        c2 = Customer("Alex Chen", "alex@example.com", 2, "Basic", 9.99, "2026-06-05")
        
        manager.add(c1)
        manager.add(c2)
    
    # 3. Persist state tracking data
    manager.save()
    
    # 4. Access states seamlessly through our clean @property abstractions
    print(f"\nTotal Customers: {manager.customer_count}")
    print(f"Active Customers: {len(manager.active_customers)}")
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()