# 🚛 IndustryGPT: Specialized Logistics & Supply Chain LLM

## 📌 Project Overview
IndustryGPT is a specialized Large Language Model (LLM) fine-tuned specifically for the Transportation and Logistics sector. It is designed to act as an expert supply chain assistant, capable of resolving complex customer queries regarding shipping delays, freight policies, packet scanning workflows, and order tracking. 

## 🎯 Key Features
*   **Industry-Specific Focus:** Fine-tuned on a hybrid dataset that merges Bitext customer support dialogues with real-world Kaggle operational data (delivery times, weather conditions, and routing status).
*   **Zero-Hallucination Knowledge Base (RAG):** Integrates a semantic vector database to pull exact company policies and procedures into the model's context window, ensuring factual responses for hub operations and exception handling.
*   **Compute Efficient:** Developed entirely within a Google Colab T4 GPU (16 GB VRAM) limit using a highly efficient 1.1-billion parameter base model.
*   **Interactive Web UI:** Features a Gradio-powered chatbot interface for real-time user testing and demonstration.

## 🛠️ Tech Stack & Architecture
*   **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
*   **Quantization:** `BitsAndBytes` (4-bit NormalFloat / NF4)
*   **Fine-Tuning Methodology:** `PEFT` / QLoRA (Low-Rank Adaptation) with a Rank (r) of 16.
*   **Training Framework:** Hugging Face `TRL` (`SFTTrainer` & `SFTConfig`)
*   **RAG Architecture:** `FAISS` (Vector Database) & `sentence-transformers` (`all-MiniLM-L6-v2`)
*   **Data Processing:** `Pandas` and Hugging Face `Datasets`
*   **Deployment:** `Gradio` UI Framework

## 📊 Dataset Configuration
The training pipeline utilizes a unified corpus built from three layers:
1.  **Customer Support Layer:** 2,000 carefully filtered rows from the *Bitext Customer Support Dataset*, discarding irrelevant account management intents to isolate shipping, delivery, and refund intents.
2.  **Operational Layer:** 1,150 synthetic conversational rows generated from the *Delivery Logistics Kaggle Dataset*, providing the model with factual routing and status data.
3.  **Company Policy Layer (Vector DB):** A custom operational FAQ dataset covering highly specific internal workflows (e.g., inbound packet scanning, plastic chain-guide maintenance protocols) injected via Retrieval-Augmented Generation.

## 📓 Master Architecture Pipeline

### Phase 1: Environment & Infrastructure Setup
* **Environment Initialization:** Install core dependencies (Hugging Face ecosystem, PyTorch, FAISS, Gradio).
* **Workspace Configuration:** Mount Google Drive and establish persistent cloud storage.
* **Directory Architecture:** Construct the project folder hierarchy, including dedicated storage for vector databases (`vector_db/faiss_index`) and model checkpoints.

### Phase 2: Data Engineering & Knowledge Curation
* **Customer Support Ingestion:** Load and filter domain-specific Bitext dialogue data.
* **Logistics Dataset Integration:** Merge with Kaggle delivery logistics data for operational context.
* **Data Sanitization:** Execute cleaning protocols (handling nulls, removing duplicates, and text normalization).
* **Company Policy Knowledge Base:** Construct a custom FAQ dataset containing specific supply chain protocols, hub scanning workflows, and fleet maintenance procedures.
* **Synthetic Retail Generation:** Generate and merge e-commerce retail conversations to bridge consumer queries with backend shipping operations.

### Phase 3: Hardware-Optimized Fine-Tuning (QLoRA)
* **Instruction Formatting:** Convert the unified master dataset into structured prompt-completion pairs.
* **Model Instantiation:** Load the `TinyLlama-1.1B` base model and `bitsandbytes` 4-bit quantization configuration.
* **Hardware Alignment:** Explicitly cast parameters to `FP16` to bypass T4 GPU `BFloat16` architecture constraints.
* **LoRA Configuration:** Inject Low-Rank Adaptation adapters into the model's attention and projection layers.
* **Training Execution:** Run the `SFTTrainer` (with GradScaler disabled for stability) to adapt the model to the logistics dataset.
* **Checkpoint Export:** Save the fine-tuned LoRA adapter weights and custom tokenizer.

### Phase 4: Retrieval-Augmented Generation (RAG)
* **Knowledge Base Embedding:** Use `sentence-transformers` to generate numerical vectors from the custom operational FAQ dataset.
* **Vector Database Creation:** Build and populate the `FAISS` index for rapid semantic similarity search.
* **Context Retrieval:** Implement the search function to fetch relevant operational policies based on user queries.
* **Context-Aware Generation:** Merge the retrieved context with the fine-tuned model (in inference mode) to generate factually grounded, hallucination-free responses.

### Phase 5: Interactive Deployment & Evaluation
* **Web UI Deployment:** Wrap the generation pipeline in a Gradio `ChatInterface` for real-time visual testing.

## 🚀 Installation & Setup
To run this project locally or in a fresh Collab environment, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/abhik99//IndustryGPT.git](https://github.com/abhik99//IndustryGPT.git)
   cd IndustryGPT
   
2. **Install Dependencies**
    ```bash

    pip install -r requirements.txt

3. **Run the Notebook**
Open the Colab/Jupyter Notebook and execute the cells sequentially. The architecture is explicitly configured to handle hardware constraints on a standard 16GB T4 GPU.

### 🔮 Future Scope

   - Dynamic API Integration: Connecting the chatbot directly to live tracking databases to provide real-time package locations and dynamic weather-delay updates.

   - Advanced RAG (Hybrid Search): Combining FAISS vector search with BM25 keyword search to accurately retrieve exact alphanumeric strings like tracking numbers and SKU codes.

   - Production Deployment: Containerizing the architecture with Docker and deploying the Gradio frontend to Hugging Face Spaces for permanent public access.
