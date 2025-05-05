import { MistralAI } from "@langchain/mistralai";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { JSONLoader } from "@langchain/community/document_loaders/fs/json";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { MistralAIEmbeddings } from "@langchain/mistralai/embeddings";
import fs from 'fs';
import 'dotenv/config';

// Initialize MistralAI LLM
const llm = new MistralAI({
  apiKey: process.env.MISTRL_API_KEY || 'your_api_key',
  model: "codestral-latest",
  temperature: 0,
  maxTokens: undefined,
  maxRetries: 2,
});

// Initialize MistralAI embeddings
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRL_API_KEY || 'your_api_key',
});

// Create a text splitter for long content
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
  separators: ["\n\n", "\n", " ", ""],
});

// Function to split array into chunks
function chunkArray(array, chunkSize) {
  const chunks = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    chunks.push(array.slice(i, i + chunkSize));
  }
  return chunks;
}

try {
  // Load JSON file directly
  const threads = JSON.parse(fs.readFileSync('stackoverflow_overlap_threads.json', 'utf-8'));
  
  // Process documents into separate question, answer, and comment documents
  const processedDocs = [];
  
  for (const thread of threads) {
    const questionId = thread.question_id;
    
    // Create question document
    const questionDoc = {
      pageContent: `Question: ${thread.title}\n\n${thread.question_body}`,
      metadata: {
        type: 'question',
        question_id: questionId,
        score: thread.score,
        tags: thread.tags,
        view_count: thread.view_count
      }
    };
    processedDocs.push(questionDoc);
    
    // Process answers if they exist
    if (thread.answers && Array.isArray(thread.answers)) {
      for (const answer of thread.answers) {
        const answerDoc = {
          pageContent: `Answer: ${answer.body}`,
          metadata: {
            type: 'answer',
            question_id: questionId,
            score: answer.score
          }
        };
        processedDocs.push(answerDoc);
      }
    }
    
    // Process comments if they exist
    if (thread.comments && Array.isArray(thread.comments)) {
      for (const comment of thread.comments) {
        const commentDoc = {
          pageContent: `Comment: ${comment.body}`,
          metadata: {
            type: 'comment',
            question_id: questionId,
            score: comment.score
          }
        };
        processedDocs.push(commentDoc);
      }
    }
  }
  
  // Split long documents if needed
  const finalDocs = [];
  for (const doc of processedDocs) {
    if (doc.pageContent.length > 1000) {
      const splitDocs = await textSplitter.splitDocuments([doc]);
      finalDocs.push(...splitDocs);
    } else {
      finalDocs.push(doc);
    }
  }
  
  // Print sample of processed documents
  console.log("\nSample of processed documents:");
  finalDocs.slice(0, 5).forEach((doc, index) => {
    console.log(`\n=== Document ${index + 1} ===`);
    console.log("Type:", doc.metadata.type);
    console.log("Question ID:", doc.metadata.question_id);
    console.log("Content:", doc.pageContent);
    console.log("=".repeat(50));
  });
  
  // Split documents into batches
  const batchSize = 100; // Adjust this number based on your needs
  const docBatches = chunkArray(finalDocs, batchSize);
  
  console.log(`\nStoring ${finalDocs.length} documents in ${docBatches.length} batches...`);
  
  // Initialize ChromaDB
  let vectorStore = null;
  
  // Process each batch
  for (let i = 0; i < docBatches.length; i++) {
    console.log(`Processing batch ${i + 1}/${docBatches.length}...`);
    if (i === 0) {
      // First batch creates the collection
      vectorStore = await Chroma.fromDocuments(
        docBatches[i],
        embeddings,
        {
          collectionName: "stackoverflow_threads",
          url: "http://localhost:8000",
        }
      );
    } else {
      // Subsequent batches add to the existing collection
      await vectorStore.addDocuments(docBatches[i]);
    }
  }
  
  console.log("\nAll documents stored in vector database");
  
  // Example: Perform a similarity search
  const results = await vectorStore.similaritySearch("What is the best way to handle overlapping threads?", 3);
  console.log("\nSimilarity search results:");
  results.forEach((result, index) => {
    console.log(`\n=== Result ${index + 1} ===`);
    console.log("Type:", result.metadata.type);
    console.log("Question ID:", result.metadata.question_id);
    console.log("Content:", result.pageContent);
    console.log("=".repeat(50));
  });
  
} catch (error) {
  console.error('Error:', error.message);
}