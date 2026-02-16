# 🌾 RSSA Field Instruments Guide (RAG)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** application designed to provide technically accurate, domain-specific insights from scientific field instrument manuals. This tool acts as a digital technical assistant, ensuring researchers have instant access to grounded data without manual searching.

---

## 🚀 Features

* **Context-Aware Q&A**: Answers are strictly grounded in provided technical manuals to prevent AI hallucinations.
* **Mathematical Precision**: Retains exact formatting for equations, scientific variables, and technical notation.
* **High-Speed Inference**: Powered by **Groq** LPUs for near-instantaneous response times.
* **Vector Search**: Utilizes **FAISS** for efficient similarity searching across segmented document chunks.
* **Transparent Sourcing**: Includes an integrated document viewer to see the exact context used for every answer.

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **LLM**: Groq (Llama 3 / Mixtral)
* **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Document Parsing**: PyPDF

---

## 📂 Project Structure

```text
├── .streamlit/
│   └── config.toml         # UI Theme and configuration
├── data/
│   └── instruments_guide/  # Source PDF manuals (Context Library)
├── streamlit_app.py        # Optimized application logic
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation
