# **Responsive Design Fix (ReDeFix)**

## Overview

This tool helps web developers automatically fix CSS layout issues by:
- Analyzing reported layout failures from input JSON
- Identifying problematic CSS properties
- Retrieving relevant StackOverflow discussions using semantic search
- Generating targeted repair suggestions using Mistral AI

## Components
1. ChromaDB Vector Store: Stores embeddings of StackOverflow questions and answers
2. BM25 Retriever: Provides keyword-based search for CSS properties
3. Ensemble Retrieval: Combines vector similarity and BM25 for optimal results
4. Mistral AI: Generates embeddings and provides CSS repair suggestions

## 📁 Structure  
Repository/  
├── index.js             # Main entry point and orchestration  
├── readJson.js          # Utility for reading layout failure data  
├── retrieval.js         # Handles similarity search and document retrieval  
├── sendImage.js         # Converts images to base64 for LLM processing  
├── rlf_full_list.json   # Input data with layout failure information  
├── repairs/             # Output directory for generated fixes  
└── package.json         # Project dependencies  

### **Prerequisites**  
Node.js (v16+)  
ChromaDB running locally  
Mistral API key  
Stack Exchange API key  
