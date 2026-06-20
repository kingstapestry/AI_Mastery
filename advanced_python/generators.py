import time
from typing import Generator, List


"""
LESSON 27: Generators & Iterators

Goal:
    Master generators - a powerful Python feature for lazy evaluation and memory efficiency.
    Extremely useful for large datasets, streaming, and ML training.
"""


# ==================== 1. BASIC GENERATOR ====================
def count_up_to(n: int) -> Generator[int, None, None]:
    """Simple generator that yields numbers from 1 to n."""
    count = 1
    while count <= n:
        yield count                     # yield pauses the function and returns the value
        count += 1


# ==================== 2. MEMORY-EFFICIENT FILE READER ====================
def read_large_file(file_path: str) -> Generator[str, None, None]:
    """
    Reads a large file line by line without loading everything into memory.
    Perfect for processing huge log files or datasets.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield line.strip()          # yield one line at a time


# ==================== 3. GENERATOR EXPRESSION ====================
def squares_generator(n: int) -> Generator[int, None, None]:
    """Generator expression example - memory efficient."""
    for i in range(n):
        yield i ** 2


# ==================== 4. INFINITE GENERATOR ====================
def infinite_counter(start: int = 0) -> Generator[int, None, None]:
    """Infinite generator - can run forever until manually stopped."""
    current = start
    while True:
        yield current
        current += 1


# ==================== 5. GENERATOR WITH SEND (Bidirectional) ====================
def echo() -> Generator:
    """
    Advanced generator that can receive values using .send().
    Useful for interactive generators or coroutines.
    """
    while True:
        received = yield                    # yield can also receive values
        print(f"Received from .send(): {received}")


# ==================== 6. PRACTICAL BATCH GENERATOR (ML Training) ====================
def batch_generator(data: List, batch_size: int) -> Generator[List, None, None]:
    """
    Yields data in batches - very useful for training ML models on large datasets.
    """
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    # Yield the last incomplete batch if any
    if batch:
        yield batch


# ==================== 7. MAIN DEMO ====================
def main():
    print("=== Generators & Iterators Demo ===\n")
    
    # Basic Generator
    print("Basic Generator:")
    for num in count_up_to(5):
        print(num, end=" ")
    print("\n")
    
    # Generator Expression
    print("Generator Expression (Squares):")
    for sq in squares_generator(5):
        print(sq, end=" ")
    print("\n")
    
    # Infinite Generator (with break)
    print("Infinite Generator (first 5 values):")
    counter = infinite_counter()
    for _ in range(5):
        print(next(counter), end=" ")
    print("\n")
    
    # Generator with .send()
    print("Generator with .send():")
    echo_gen = echo()
    next(echo_gen)                          # Prime the generator
    echo_gen.send("Hello")
    echo_gen.send("World")
    print()
    
    # Batch Generator (ML-style)
    print("Batch Generator Example:")
    data = list(range(20))
    for batch in batch_generator(data, batch_size=6):
        print(batch)
    
    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()


# ==================== SUMMARY ====================
"""
Key Takeaways from Generators:

- Generators use `yield` instead of `return`
- They are lazy: values are produced only when requested (memory efficient)
- Perfect for large datasets, file processing, streaming, and ML batching
- `next()` is used to get the next value
- Can be bidirectional with `.send()`
"""