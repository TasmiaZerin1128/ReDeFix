import { Chroma } from '@langchain/community/vectorstores/chroma';
import { ChromaClient } from "chromadb";
import { MistralAIEmbeddings } from '@langchain/mistralai';
import { EnsembleRetriever } from "langchain/retrievers/ensemble";
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { Document } from 'langchain/document';
import 'dotenv/config';

// Initialize environment
const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';

const embeddings = new MistralAIEmbeddings({
    apiKey: apiKey,
    model: "mistral-embed"
});

async function retrieve(collectionDBName, cssProperties) {
  try {
    // Connect to existing ChromaDB collection
    console.log("Connecting to ChromaDB for retrieval");
    const client = new ChromaClient({
      url: "http://localhost:8000",
    });

    let vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: collectionDBName,
      client,
    });

    const rawCollection = await client.getOrCreateCollection({
      name: collectionDBName,
    });
    const { documents: texts, metadatas } = await rawCollection.get({
      include: ["documents", "metadatas"],
    });

    const docs = texts.map(
      (txt, i) => new Document({ pageContent: txt, metadata: metadatas[i] })
    );

    console.log("Number of documents in collection:", docs.length);

    const k_retriever = 5;

    const bm25Retriever = new BM25Retriever({
      docs: docs.filter((d) => d.metadata.type === "question"),
      k: k_retriever,
    });

    const vectorstoreRetriever = vectorStore.asRetriever({
      k: k_retriever,
      filter: { type: "question" },
    });

    const ensembleRetriever = new EnsembleRetriever({
      retrievers: [bm25Retriever, vectorstoreRetriever],
      weights: [0.6, 0.4],
    });

    const questions = await ensembleRetriever.invoke(cssProperties.join(" "));
    console.log(questions.length + " Questions retrieved");

    for (const question of questions) {
      const questionId = question.metadata.question_id;
      const answers = docs.filter(
        (d) =>
          d.metadata.type === "answer" && d.metadata.question_id === questionId
      );
      question.answers = answers;
      const comments = docs.filter(
        (d) =>
          d.metadata.type === "comment" && d.metadata.question_id === questionId
      );
      question.comments = comments;
    }

    return questions;
  } catch (error) {
    console.error("Error during retrieval:", error);
  }
}

export default retrieve;