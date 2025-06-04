import { Chroma } from '@langchain/community/vectorstores/chroma';
import { ChromaClient } from "chromadb";
import { MistralAIEmbeddings } from '@langchain/mistralai';
import { EnsembleRetriever } from "langchain/retrievers/ensemble";
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { Document } from 'langchain/document';
import 'dotenv/config';

// Initialize environment
const apiKey = process.env.MISTRAL_API_KEY || 'your_api_key';

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

    const all_docs = texts.map(
      (txt, i) => new Document({ pageContent: txt, metadata: metadatas[i] })
    );

    console.log("Number of documents in collection:", all_docs.length);

    const k_retriever = 5;

    const bm25Retriever = new BM25Retriever({
      docs: all_docs.filter((d) => d.metadata.type === "question"),
      k: k_retriever,
    });

    const vectorstoreRetriever = vectorStore.asRetriever({
      k: k_retriever,
      filter: { type: "question" },
    });

    const ensembleRetriever = new EnsembleRetriever({
      retrievers: [bm25Retriever, vectorstoreRetriever],
      weights: [0.8, 0.2],
    });

    console.log(cssProperties.join(" "));
    const questions = await ensembleRetriever.invoke(cssProperties.join(" "));
    console.log(questions.length + " Questions retrieved");

    for (const question of questions) {
      const questionId = question.metadata.question_id;
      const answers = all_docs.filter(
        (d) =>
          d.metadata.type === "answer" && d.metadata.question_id === questionId
      );
      question.answers = answers;
      const comments = all_docs.filter(
        (d) =>
          d.metadata.type === "comment" && d.metadata.question_id === questionId
      );
      question.comments = comments;
    }

    let SO_discussions_formatted = questions.map((q) => {
      return {
        question: q.pageContent,
        answers: q.answers.map((a) => a.pageContent),
        comments: q.comments.map((c) => c.pageContent),
      }
    })

    return SO_discussions_formatted;
  } catch (error) {
    console.error("Error during retrieval:", error);
  }
}

export default retrieve;