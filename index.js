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

try {
    const threads = JSON.parse(fs.readFileSync('stackoverflow_overlap_threads.json', 'utf-8'));

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1500,
        chunkOverlap: 200,
        separators: ["\n\n", "\n", " ", ""],
    })
    const SO_collection = threadToDocument(threads, textSplitter);
    const documents = processDocuments(SO_collection, textSplitter);
    storeInVectorDB(documents);
} catch (error) {
    console.error("Error processing JSON file:", error);
}


function threadToDocument(threads) {
    const SO_collection = [];

    for (const thread of threads){
        const questionId = thread.question_id;
        const title = thread.title;
        const body = thread.question_body;

        const questionDoc = {
            pageContent: `Question: ${title}\n\n${body}`,
            metadata: {
                type: "question",
                rlf: "type",
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

        if (thread.comments) {
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
    }
    return SO_collection;
} 

async function processDocuments(documents, textSplitter) {
    console.log(`Splitting ${documents.length} documents...`);
    const finalDocs = [];
    
    for (const doc of documents) {
        try {
            const splitDocs = await textSplitter.splitDocuments([doc]);
            finalDocs.push(...splitDocs);
        } catch (error) {
            console.error("Error splitting document:", error.message);
        }
    }
    
    console.log(`Split into ${finalDocs.length} chunks`);
    return finalDocs;
}

async function storeInVectorDB(finalDocs) {
    // Initialize embeddings
    const embeddings = new MistralAIEmbeddings({
        apiKey: apiKey,
        model: "mistral-embed"
    });
    
    // Store in ChromaDB with batching
    console.log("Storing documents in ChromaDB...");
    
    try {
        const initDocs = finalDocs.slice(0, 1);
        let vectorStore = await Chroma.fromDocuments(
            initDocs,
            embeddings,
            { 
                collectionName: "stackoverflow_data",
                url: 'http://localhost:8000',
                collectionMetadata: {
                    "hnsw:space": "cosine"
                }
            }
        );
        
        const remainingDocs = finalDocs.slice(1);
        const batchSize = 200;
        console.log(`Processing ${remainingDocs.length} remaining documents in batches of ${batchSize}...`);
        
        for (let i = 0; i < remainingDocs.length; i += batchSize) {
            try {
                const batch = remainingDocs.slice(i, i + batchSize);
                console.log(`Processing batch ${Math.floor(i/batchSize) + 1} of ${Math.ceil(remainingDocs.length/batchSize)}`);
                
                await vectorStore.addDocuments(batch);
                console.log(`Processed batch of ${batch.length} documents`);
                
            } catch (batchError) {
                console.error(`Error processing batch: ${batchError.message}`);
                console.log("Falling back to processing one document at a time...");
                
                const batchStart = i;
                const batchEnd = Math.min(i + batchSize, remainingDocs.length);
                
                for (let j = batchStart; j < batchEnd; j++) {
                    try {
                        await vectorStore.addDocuments([remainingDocs[j]]);
                        console.log(`Processed document ${j + 1} of ${remainingDocs.length}`);
                    } catch (docError) {
                        console.error(`Could not process document ${j + 1}: ${docError.message}`);
                    }
                }
            }
            
            // Add a small delay between batches to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        console.log("Successfully stored all documents in ChromaDB");
        return vectorStore;
        
    } catch (error) {
        console.error("Error initializing ChromaDB:", error);
        throw error;
    }
}
