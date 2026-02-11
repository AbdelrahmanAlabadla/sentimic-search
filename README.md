# 🔍💬 Semantic Search Engine – NLP with Sentence Transformers
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)  
![Pandas](https://img.shields.io/badge/Pandas-2.1-lightgrey?logo=pandas)  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2-orange?logo=scikit-learn)  
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2-purple)  
![License](https://img.shields.io/badge/License-MIT-green)  

End-to-end **NLP project** | **Semantic search** using sentence embeddings  
Built by Abdelrahman Alabadla • Feb 2026  
Email: abdelrahmanalabadla@gmail.com  

---

## Overview
This project implements a **semantic search engine** that retrieves the most **contextually relevant sentences** from a dataset using **sentence embeddings**. Unlike keyword search, it understands meaning, making it ideal for customer support, FAQs, and document search.

- **Problem Type:** Semantic Search / Information Retrieval  
- **Model:** `all-MiniLM-L6-v2` (Sentence Transformers)  
- **Input:** Query text  
- **Output:** Top-K semantically similar sentences with categories and similarity scores  

---

## Dataset
- **Source:** CSV file with sentences and categories (`Sentimic_Search.csv`)  
- **Records:** 381 sentences  
- **Features:**  
  - `Sentence`: The text to search  
  - `Category`: Label or type of sentence  

---

## Data Preprocessing
- Remove extra spaces and normalize whitespace  
- Lowercase text (optional depending on model)  
- Remove URLs and digits  
- Remove punctuation (keep apostrophes)  
- Remove emojis  

---

## Modeling
- **Sentence Embeddings:** Convert sentences to 384-dimensional vectors (`all-MiniLM-L6-v2`)  
- **Normalization:** L2-normalize embeddings  
- **Similarity Metric:** Cosine similarity  
- **Thresholding:** Only return sentences above a similarity score of `0.6`  

**Embedding shape example:** `(381, 384)`  

---

## Sample Query
```python
engine.search("I need to update my payment method")


Similarity results:
[Payment Issues] How do I update my payment method - Score: 0.9761
[Billing] How do I add a new payment method - Score: 0.7805
[Billing] I want to change my payment details - Score: 0.7264
[Billing] How do I remove an old payment method - Score: 0.6938
[Billing] How do I update my credit card - Score: 0.6686
```
---

## Features / Usage

- Preprocesses sentences automatically for cleaner embeddings  
- Supports saving and loading embeddings to avoid recomputation  
- Top-K search with optional threshold filtering

