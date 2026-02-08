# Financial Report Summarizer Agent

This project implements an intelligent document ingestion pipeline and an AI Retrieval Agent using the Google Agent Development Kit (ADK). It analyzes complex financial earnings reports (like PDFs or slides) including tables and charts, making it easy to interrogate financial data via an interactive analyst widget.

## Prerequisites

- **Python 3.10+**
- A **Google Cloud Platform (GCP)** project with billing enabled.
- The **Vertex AI API** must be enabled in your GCP project.
- Appropriate IAM roles (e.g., Vertex AI User) assigned to your service account or user credentials if running locally.

### Models Used

This architecture leverages two cutting-edge models deployed on Vertex AI:
- **`gemini-2.5-flash`**: Used for multimodal contextualization (extracting rich descriptions directly from parsed chart images) and driving the reasoning engine of the ADK `LlmAgent`.
- **`text-embedding-005`**: Used to embed semantic Markdown chunks into the ChromaDB vector database.

---

## 🚀 Setup Instructions

1. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install google-adk google-genai chromadb docling python-dotenv langchain-text-splitters
   ```

3. **Configure Environment Variables:**
   A template file `.env.example` has been provided. Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
   
   Ensure your `.env` contains the required GCP attributes:
   ```env
   GOOGLE_CLOUD_PROJECT=gcpsaptesting
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_GENAI_USE_VERTEXAI=1
   ```

---

## 🗄️ Running Data Ingestion

The `ingest.py` script powers the pipeline. It recursively scans the `earnings/` directory for PDFs, safely extracts charts to `earnings/image_cache/`, analyzes them with Gemini 2.5 Flash, formats everything into Semantic Markdown, and embeds the output into ChromaDB.

*Make sure your financial PDFs are located inside `earnings/<quarter-identifier>/` (for example, `earnings/Q1-2025/`).*

```bash
# Activate your environment if you haven't already
source .venv/bin/activate

# Run the ingestion pipeline
python ingest.py
```

*Note: Depending on the size of the PDFs and the number of charts, parsing via `docling` and processing image descriptions may take several minutes.*

---

## 🤖 Running the Financial Analyst Agent

You can interact with the analytical agent in two ways via the Google ADK:

### Option A: ADK Terminal Chat
Execute the following to talk with the agent directly in your terminal:
```bash
adk run .
```

### Option B: ADK Web UI (Recommended)
Launch a local web application that allows you to easily view the generated image and markdown charts in real-time.
```bash
adk web .
```
Open your browser to `http://localhost:8000`.

---

## 💬 Sample Prompts

Once the agent is running, try asking contextually heavy financial questions. Here are a few examples you can use:

1. **Chart Retrieval (Revenues/Income):**
   > *"Show me the Google Services Revenues and Operating Income chart for Q1 2025."*

2. **Historical Number Extraction:**
   > *"What is the Operating income in Q1 2024?"*

3. **Complex Segment Analysis:**
   > *"What was the Google Cloud revenue for Q1 2025? Please provide the chart."*

*Because of our precise chunking strategy, the agent will gracefully return exact figures alongside Markdown links to the exact visual charts extracted during ingestion!*
