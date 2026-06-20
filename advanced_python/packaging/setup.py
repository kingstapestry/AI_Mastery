"""
setup.py

Makes your package installable with `pip install -e .`
"""

from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1.0",
    packages=find_packages(),
    description="A sample package for advanced Python learning",
    author="King",
    python_requires=">=3.8",
)