"""
LESSON 32: Virtual Environments, Dependencies & Professional Project Setup

Goal:
    Learn how to manage projects professionally — a critical skill for real AI Engineering work.
"""

# ==================== 1. VIRTUAL ENVIRONMENT MANAGEMENT ====================
"""
Virtual Environment Commands:

Creation:
    python -m venv venv

Activation:
    Windows: venv\Scripts\activate
    Mac/Linux: source venv/bin/activate

Deactivation:
    deactivate

Why use venv?
- Isolates project dependencies
- Prevents version conflicts
- Makes projects reproducible
"""


# ==================== 2. REQUIREMENTS.TXT MANAGEMENT ====================
"""
Best Practices for requirements.txt:

1. Generate it:
   pip freeze > requirements.txt

2. Install from it:
   pip install -r requirements.txt

3. Use specific versions when needed:
   pandas==2.0.3
   scikit-learn>=1.3.0

4. Separate environments:
   requirements.txt (production)
   requirements-dev.txt (development)
"""


# ==================== 3. RECOMMENDED PROJECT STRUCTURE ====================
"""
Recommended Structure for AI/ML Projects:

AI_Project/
├── src/                    # Main source code
│   ├── __init__.py
│   ├── models/
│   ├── data/
│   └── utils/
├── tests/                  # Unit tests
├── notebooks/              # Jupyter experiments
├── data/                   # Raw and processed data
├── models/                 # Saved models
├── logs/                   # Log files
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── .gitignore
└── README.md
"""


# ==================== 4. .GITIGNORE BEST PRACTICES ====================
"""
Recommended .gitignore for Python/ML Projects:

# Virtual Environments
venv/
env/
.venv/

# Python cache
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Data & Models
data/raw/
models/*.pkl
*.h5

# Logs
logs/
*.log

# Environment variables
.env

# OS files
.DS_Store
Thumbs.db
"""


# ==================== 5. SETUP.PY EXAMPLE ====================
"""
setup.py (Basic Version)

from setuptools import setup, find_packages

setup(
    name="my_ai_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
"""


# ==================== 6. PRACTICAL DEMO ====================
def main():
    print("=== Professional Project Setup Demo ===\n")
    
    print("1. Virtual Environment Commands:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate          # Windows")
    print("   pip install -r requirements.txt")
    
    print("\n2. Good practices you should follow:")
    print("   - Always use virtual environments")
    print("   - Keep requirements.txt updated")
    print("   - Use a clean project structure")
    print("   - Document everything in README.md")
    
    print("\n=== Setup Best Practices Applied ===")


if __name__ == "__main__":
    main()


# ==================== SUMMARY ====================
"""
Key Takeaways:

- Virtual environments isolate your projects
- requirements.txt makes projects reproducible
- Clean project structure improves maintainability
- Good setup = easier collaboration and deployment
- These practices are expected at professional AI companies
"""