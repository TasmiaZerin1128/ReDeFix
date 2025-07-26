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
├── src/                # Source code directory  
│   ├── index.js        # Main entry point and orchestration  
│   ├── readJson.js     # Utility for reading layout failure data  
│   ├── retrieval.js    # Handles similarity search and document retrieval  
│   ├── sendImage.js    # Converts images to base64 for LLM processing  
├── Input/              # Input data directory  
│   ├── rlf_full_list.json   # Input data with layout failure information
|   ├── Knowledge_Base  # Stack Exchange QA Retriever Script and output QA jsons
├── repairs/            # Output directory for generated fixes  
├── package.json        # Project dependencies  
└── README.md           # Project documentation

### **Prerequisites**  
Node.js (v16+)  
ChromaDB running locally  
Mistral API key  
Stack Exchange API key

## Installation Guide

Follow these steps to set up and run the project:

### Step 1: Clone the Repository
```bash
git clone https://github.com/TasmiaZerin1128/ReDeFix.git
cd ReDeFix
```

### Step 2: Install Dependencies
Make sure you have Node.js (v16+) installed. Then, run:
```bash
npm install
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the root directory and add the following keys:
```env
MISTRAL_API_KEY=your_mistral_api_key
STACK_EXCHANGE_API_KEY=your_stack_exchange_api_key
```
Replace `your_mistral_api_key` and `your_stack_exchange_api_key` with your actual API keys.
For both keys, you will have to create a Mistral account and a Stack Exchange Developers account.

### Step 4: Start ChromaDB
Ensure ChromaDB is running locally. If not installed, install it via pip:
```bash
pip install chromadb
chromadb run
```

### Step 5: Run the Tool
Execute the main script from the src folder:
```bash
node index.js
```

### Step 6: View Results
Check the `repairs/` directory for generated fixes.
