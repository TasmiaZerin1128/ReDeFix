import { Mistral } from '@mistralai/mistralai';
import 'dotenv/config';
import { JSONLoader } from 'langchain/document_loaders/fs/json';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { MistralAIEmbeddings } from '@langchain/mistralai';
import fs from 'fs';

const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';
console.log('API Key:', apiKey);

const client = new Mistral({apiKey: apiKey});

const threads = JSON.parse(fs.readFileSync('stackoverflow_overlap_threads.json', 'utf-8'));

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
    separators: ["\n\n", "\n", " ", ""],
})

const SO_collection = [];

for (const thread of threads){
    const questionId = thread.question_id;
    const title = thread.title;
    const body = thread.question_body;

    const questionDoc = {
        pageContent: `Question: ${title}\n\n${body}`,
        metadata: {
          type: "question",
          question_id: questionId,
          title: title,
          score: thread.score || 0,
          tags: (thread.tags || []).join(','),
          link: thread.link || '',
          view_count: thread.view_count || 0,
          answer_count: thread.answer_count || 0,
          comment_count: thread.comment_count || 0
        }
    }
    SO_collection.push(questionDoc);

    for (const answer of thread.answers){
        const answerDoc = {
            pageContent: `Answer: ${answer.body}`,
            metadata: {
              type: 'answer',
              question_id: questionId,
              score: answer.score
            }
        };
        SO_collection.push(answerDoc);
    }

    for (const comment of thread.comments) {
        const commentDoc = {
          pageContent: `Comment: ${comment.body}`,
          metadata: {
            type: 'comment',
            question_id: questionId,
          }
        };
        SO_collection.push(commentDoc);
    }
}

async function processDocuments() {
    const finalDocs = [];
    for (const doc of SO_collection) {
    if (doc.pageContent.length > 1000) {
        const splitDocs = await textSplitter.splitDocuments([doc]);
        finalDocs.push(...splitDocs);
    } else {
        finalDocs.push(doc);
    }
    }
    console.log("\nSample of processed documents:");
    finalDocs.slice(0, 5).forEach((doc, index) => {
        console.log(`\n=== Document ${index + 1} ===`);
        console.log("Type:", doc.metadata.type);
        console.log("Question ID:", doc.metadata.question_id);
        console.log("Content:", doc.pageContent);
        console.log("=".repeat(50));
    });

    const embeddings = new MistralAIEmbeddings({
        apiKey: apiKey,
        model: "mistral-embed"
    });
    
    console.log("Storing documents in ChromaDB...");
    const vectorStore = await Chroma.fromDocuments(
        finalDocs,
        embeddings,
        { collectionName: "stackoverflow_data" }
    );
    
    console.log("Successfully stored documents in ChromaDB");
    return vectorStore;
}

processDocuments().catch(console.error);
