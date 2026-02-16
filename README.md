# RAG-application
🌾 RSSA Field Instruments Guide (RAG)
An advanced Retrieval-Augmented Generation (RAG) application designed to provide technically accurate, domain-specific insights from scientific field instrument manuals. This tool leverages high-performance LLMs and vector embeddings to act as a digital technical assistant for field researchers.

🚀 Features
Context-Aware Q&A: Answers are strictly grounded in provided technical manuals to prevent hallucinations.

Mathematical Precision: Retains exact formatting for equations and scientific variables.

High-Speed Inference: Powered by Groq LPUs for near-instant response times.

Vector Search: Utilizes FAISS for efficient similarity searching across document chunks.

Transparent Sourcing: Provides an expandable view of the exact document segments used to generate each response.

🛠️ Tech Stack
Frontend: Streamlit

Orchestration: LangChain

LLM: Groq (Llama 3 / Mixtral)

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Vector Store: FAISS (Facebook AI Similarity Search)

Document Parsing: PyPDF

📂 Project Structure
Plaintext
├── .streamlit/
│   └── config.toml         # UI Theme configuration
├── data/
│   └── instruments_guide/  # Technical PDF manuals (Source Context)
├── streamlit_app.py        # Main application logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
🔧 Local Setup & Installation
Clone the Repository:

Bash
git clone https://github.com/your-username/rssa-instruments-rag.git
cd rssa-instruments-rag
Install Dependencies:

Bash
pip install -r requirements.txt
Environment Variables:
Create a .env file in the root directory and add your API keys:

Code snippet
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
Run the Application:

Bash
streamlit run streamlit_app.py
☁️ Deployment (Streamlit Cloud)
To deploy this application to the cloud:

Push your code to GitHub (ensure data/ folder contains your PDFs).

Connect your repository to Streamlit Community Cloud.

In Advanced Settings, add your GROQ_API_KEY and HF_TOKEN to the Secrets section.

📝 Usage Note
The system is configured as a closed-domain assistant. It is instructed to use only the provided context; if a question cannot be answered using the uploaded manuals, the assistant will explicitly state that the information is unavailable.
