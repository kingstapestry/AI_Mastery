import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
print("Array:", arr1)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print("\nMatrix:\n", matrix)

# Useful properties
print("Shape:", matrix.shape)
print("Data type:", matrix.dtype)
print("Size:", matrix.size)

# Basic operations
print("\nArray + 10:", arr1 + 10)
print("Array * 2:", arr1 * 2)
print("Sum:", arr1.sum())
print("Mean:", arr1.mean())
print("Max:", arr1.max())


# ==================== EXERCISES ====================
# Exercise 9.1:
# Create a 1D array with numbers from 10 to 20 (use np.arange)

# Exercise 9.2:
# Create a 3x3 matrix filled with zeros, then fill the diagonal with 5s
# (Hint: use np.zeros() and np.eye() or manual indexing)

# Exercise 9.3:
# Create two arrays: a = [1,2,3,4] and b = [5,6,7,8]
# Print their element-wise sum, product, and difference

# Write your code for Exercises 9.1 - 9.3 here:

# 1D array from 10 to 20
numArr2  = np.arange(10, 21)
print(numArr2)

# 3x3 matrix of zeros, diagonals with 5
arr3 = np.zeros((3, 3))
np.fill_diagonal(arr3, 5)
print(arr3)

# Operations of 2 arrays simulataneously by elements
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("Sum:", a + b)
print("Product:", a * b)
print("Difference:", a - b)