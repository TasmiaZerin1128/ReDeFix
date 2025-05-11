import { Chroma } from '@langchain/community/vectorstores/chroma';
import { MistralAIEmbeddings } from '@langchain/mistralai';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import 'dotenv/config';

// Initialize environment
const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';

async function retrieve(collectionDB, cssProperties) {
  try {

    // Connect to existing ChromaDB collection
    console.log("Connecting to ChromaDB for retrieval");

    const questions = await collectionDB.similaritySearch(
      "",
      100,
      { type: "question" },
    );

    // const bm25Docs = questions.ids.map((id, index) => 
    //   new Document({
    //     pageContent: questions.documents[index],
    //     metadata: questions.metadatas[index],
    //   })
    // );


    const bm25Retriever = new BM25Retriever({
      documents: questions,
      // Optional: Customize tokenizer for code-aware splitting
      tokenizer: (text) => text.split(/(?<=[^\w-])|(?=[^\w-])/), 
    });

    const similarQuestions = await bm25Retriever.invoke(cssProperties.join(" "));

    console.log("Questions retrieved:", similarQuestions);
    return questions;

  } catch (error) {
    console.error("Error during retrieval:", error);
  }
}

// retrieve('SO_collision', ['padding', 'display'])

export default retrieve;