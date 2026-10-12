# 🚛 IndustryGPT / TransportGPT: Specialized Logistics & Supply Chain LLM

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23326308-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.23326308)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%26%20TRL-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![TinyLlama](https://img.shields.io/badge/Base%20LLM-TinyLlama--1.1B--Chat-black?style=for-the-badge)](https://huggingface.co/TinyLlama)
[![QLoRA](https://img.shields.io/badge/PEFT-QLoRA%20(Rank%2016)-8A2BE2?style=for-the-badge)](https://github.com/huggingface/peft)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS%20Dense%20Index-00599C?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A high-efficiency domain-specialized Large Language Model (LLM) fine-tuned specifically for the Transportation, Freight, and Logistics sector. Engineered to resolve complex supply chain inquiries, delivery exception protocols, warehouse status lookups, and shipping dispute policies with zero hallucinations by coupling QLoRA parameter-efficient fine-tuning with FAISS dense vector retrieval.

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture](#-system-architecture)
- [Key Engineering Highlights](#-key-engineering-highlights)
- [Hybrid Dataset Strategy](#-hybrid-dataset-strategy)
- [Fine-Tuning Methodology (QLoRA)](#-fine-tuning-methodology-qlora)
- [Vector Retrieval & RAG Grounding](#-vector-retrieval--rag-grounding)
- [Repository Structure](#-repository-structure)
- [Installation & Getting Started](#-installation--getting-started)
- [Author & Connect](#-author--connect)

---

## 🚀 Executive Summary
Generic open-weight LLMs frequently fail or hallucinate when confronted with specialized domain jargon—such as Bill of Lading (BOL), customs detention fees, demurrage rates, hazardous materials transit rules, and carrier handoffs.

**TransportGPT / IndustryGPT** delivers a lightweight, edge-deployable solution:
- **Base Architecture:** `TinyLlama-1.1B-Chat-v1.0` fine-tuned on a single consumer GPU (Google Colab T4 16GB VRAM) at zero infrastructure cost.
- **Quantization & PEFT:** 4-bit NormalFloat (NF4) quantization via `BitsAndBytes` combined with Low-Rank Adaptation (LoRA: Rank 16, Alpha 32) touching only ~1.2% of total parameters.
- **RAG Augmentation:** FAISS dense index populated with logistics standard operating procedures (SOPs) and grounded via `all-MiniLM-L6-v2` dense embeddings.
- **Interactive Serving:** Real-time Gradio web application for dispatchers and freight support teams.

---

## 🏗️ System Architecture

```text
[ User Query: "Driver delayed at Port Newark - customs hold protocol?" ]
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │          FAISS Vector Search (Retriever)               │
      │  • Encodes query using all-MiniLM-L6-v2 embeddings     │
      │  • Top-k cosine similarity search in knowledge_base    │
      │  • Extracts verified Port Customs SOP & escalation SLA │
      └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │       Prompt Augmentation & Context Injection          │
      │  System Prompt + Retrieved Document Chunks + Query     │
      └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │          Fine-Tuned TinyLlama-1.1B (PEFT/QLoRA)        │
      │  • 4-Bit NormalFloat Quantized Weights                 │
      │  • LoRA Adapter Weights (Rank 16, Alpha 32)            │
      │  • Autoregressive Token Generation                     │
      └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
      [ Fact-Checked, Domain-Accurate Logistics Resolution ]
```

---

## 🎯 Key Engineering Highlights
- **Zero-Cost Colab T4 Optimization:** Fully trained and reproducible on a single 16GB VRAM T4 instance without OOM errors.
- **Double Quantization:** Employed BitsAndBytes NF4 with nested double-quantization to shrink memory footprint below 3.2 GB during training and under 1.8 GB during inference.
- **Deterministic RAG Grounding:** The FAISS index acts as an anti-hallucination barrier, injecting up-to-date warehouse policies and shipment SLAs.
- **Interactive UI:** Built-in Gradio chat interface with temperature and top-p sampling controls.

---

## 📊 Hybrid Dataset Strategy
The model was fine-tuned on a curated hybrid dataset designed to capture conversational phrasing alongside technical freight terminology:
1. **Bitext Customer Support Dataset:** 27,000 multi-turn support interactions establishing professional escalation tone.
2. **Delivery & Logistics Operational Data:** 3,150 specialized transport question-answer pairs covering BOL processing, freight damage claims, cold-chain temperature deviations, and fleet tracking.

---

## 🧠 Fine-Tuning Methodology (QLoRA)

```python
# LoRA Configuration Specification
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

- **Quantization:** `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`.
- **Training Engine:** Hugging Face `SFTTrainer` (TRL).
- **Batch Size:** 4 with gradient accumulation steps of 4 (effective batch size 16).
- **Optimizer:** `paged_adamw_8bit` with cosine learning rate scheduler.

---

## 📁 Repository Structure
```text
├── TransportGPT.ipynb                                              # Complete end-to-end training & evaluation notebook
├── requirment.txt                                                  # Python environment dependencies
├── data/
│   └── raw/
│       ├── Bitext_Sample_Customer_Support_Training_Dataset_27K.csv # Support dialogue base
│       └── Delivery_Logistics.csv                                  # Freight and shipping domain Q&A
└── vector_db/
    └── faiss_index/
        ├── industry_faq.faiss                                      # Serialized FAISS vector index
        └── knowledge_base.csv                                      # Ground-truth SOP documentation
```

---

## 🛠️ Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/abhik99/TransportGPT.git
cd TransportGPT
```

### 2. Install Dependencies
```bash
pip install -r requirment.txt
```

### 3. Run Inference or Open Notebook
Launch `TransportGPT.ipynb` on Google Colab (GPU Runtime T4) or local Jupyter Lab to run training or spin up the Gradio user interface.

---

## 📄 Academic Research Paper
This project is backed by peer-reviewed published academic research:
* **Title:** *Resource-Efficient Fine-Tuning of Compact LLMs for Supply Chain Operations: A Hybrid QLoRA and RAG Approach*
* **Author:** Abhishek Kumar (Master's in CS: Artificial Intelligence and Machine Learning, Woolf University)
* **Permanent DOI:** [10.5281/zenodo.23326308](https://doi.org/10.5281/zenodo.23326308)
* **Citation (BibTeX):**
```bibtex
@article{kumar2026resource,
  title={Resource-Efficient Fine-Tuning of Compact LLMs for Supply Chain Operations: A Hybrid QLoRA and RAG Approach},
  author={Kumar, Abhishek},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.23326308},
  url={https://doi.org/10.5281/zenodo.23326308}
}
```

---

## 👨‍💻 Author & Connect

**Abhishek Kumar (Abhi)**  
*Data Scientist & Analytics Engineer*

- 🌐 **Portfolio:** [abhicodes.online](https://abhicodes.online)
- 💼 **LinkedIn:** [linkedin.com/in/abhik99](https://linkedin.com/in/abhik99)
- 🐙 **GitHub:** [@abhik99](https://github.com/abhik99)
- 📧 **Email:** [akdevil27boy@gmail.com](mailto:akdevil27boy@gmail.com)

---
*Distributed under the MIT License.*
