# AI Customer Support Agent

A production-oriented AI-powered customer support system built with Python.

This project aims to build an intelligent support agent capable of understanding customer queries, retrieving relevant information from a knowledge base, and generating helpful responses using Retrieval-Augmented Generation (RAG) and agentic workflows.

---

## Project Goals

- Build a real-world AI customer support system
- Implement Retrieval-Augmented Generation (RAG)
- Create an agent that can use tools and reason
- Design clean, scalable, and maintainable architecture
- Demonstrate production-ready practices (logging, configuration, testing, API)

---

## Features (Planned)

- Document ingestion (PDF, Markdown, Text)
- Vector-based knowledge retrieval
- Conversational memory
- Tool-calling agent
- Confidence-based escalation
- FastAPI backend
- Conversation history storage
- Logging and error handling

---

## Tech Stack

- **Language:** Python 3.11+
- **LLM Framework:** LangChain
- **Vector Database:** Chroma
- **API:** FastAPI
- **Embeddings:** OpenAI / compatible providers
- **Document Loading:** pypdf, LangChain document loaders

---

## Project Structure

```bash
ai-customer-support/
├── src/
│   ├── agent/           # Agent logic, tools, memory
│   ├── knowledge/       # Document ingestion & retrieval
│   ├── api/             # FastAPI routes
│   └── utils/           # Logging, config, helpers
├── data/
│   ├── raw/             # Original documents
│   └── processed/       # Processed data
├── tests/
├── notebooks/
├── scripts/
├── config/
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/kingstapestry/ai-customer-support.git
cd ai-customer-support

python -m venv venv
source venv/bin/activate    # Mac/Linux
# or 
venv\Scripts\activate       # Windows

pip install -r requirements.txt

cp .env.example .env
```

---

## Roadmap

- Project Structure ✅
- Knowledge base & RAG pipeline ✅
- Basic conversational agent
- Tool-calling capabilities
- FastAPI integration
- Conversational memory
- Evaluation & logging
- Deployment

---

## Author

**@kingstapestry**
Aspiring AI Engineer | Building production-grade AI Systems

Last Updated: August 2026