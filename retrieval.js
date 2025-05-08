import { Chroma } from '@langchain/community/vectorstores/chroma';
import { MistralAIEmbeddings } from '@langchain/mistralai';
import 'dotenv/config';

// Initialize environment
const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';

async function simpleQuery() {
  try {
    // Initialize embeddings
    const embeddings = new MistralAIEmbeddings({
      apiKey: apiKey,
      model: "mistral-embed"
    });

    // Connect to existing ChromaDB collection
    console.log("Connecting to ChromaDB...");
    const vectorStore = new Chroma(
      embeddings,
      { 
        collectionName: "stackoverflow_data",
        url: 'http://localhost:8000'
      }
    );
    
    // Perform a simple search
    const query = "How to fix CSS list bullets overlapping with floated elements";
    console.log(`Searching for: "${query}"`);
    
    const results = await vectorStore.similaritySearch(query, 3);
    
    // Display results
    console.log("\n=== SEARCH RESULTS ===\n");
    results.forEach((doc, i) => {
      console.log(`\n----- Result ${i+1} -----`);
      console.log(`Type: ${doc.metadata.type}`);
      
      if (doc.metadata.type === 'question') {
        console.log(`Title: ${doc.metadata.title}`);
      } else if (doc.metadata.type === 'answer') {
        console.log(`Question ID: ${doc.metadata.question_id}`);
      }
      
      // Show content preview
      console.log("\nContent Preview:");
      console.log(doc.pageContent);
      console.log("-".repeat(40));
    });
  } catch (error) {
    console.error("Error during retrieval:", error);
  }
}

// Run the query
simpleQuery();