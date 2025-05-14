import { ChatMistralAI } from "@langchain/mistralai";
import 'dotenv/config';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { MistralAIEmbeddings } from '@langchain/mistralai';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import fs from 'fs';
import retrieve from "./retrieval.js";
import encodeImage from "./sendImage.js";

const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';
console.log('API Key:', apiKey);

const model = new ChatMistralAI({
    model: "pixtral-12b",
    apiKey: apiKey
});

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1500,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", ""],
});

const embeddings = new MistralAIEmbeddings({
    apiKey: apiKey,
    model: "mistral-embed"
});

let collision_collection = 'SO_collision';
let protrusion_collection = 'SO_protrusion';

let collision_db = null;

async function main() {
    try {
        const collision_threads = JSON.parse(fs.readFileSync('stackoverflow_collision_threads.json', 'utf-8'));
        // const protrusion_threads = JSON.parse(fs.readFileSync('stackoverflow_overflow_threads.json', 'utf-8'));
        // const viewport_protrusion_threads = JSON.parse(fs.readFileSync('stackoverflow_viewport_protrusion_threads.json', 'utf-8'));
        // const wrapping_threads = JSON.parse(fs.readFileSync('stackoverflow_wrapping_threads.json', 'utf-8'));

        collision_db = await create_knowledge_base(collision_threads, collision_collection);
        // protrusion_db = create_knowledge_base(protrusion_threads);
        // viewport_protrusion_db = create_knowledge_base(viewport_protrusion_threads);
        // wrapping_db = create_knowledge_base(wrapping_threads);
        

    } catch (error) {
        console.error("Error processing JSON file:", error);
    }

    const retrieveDocs = await retrieve(collision_collection, ['margin-bottom']);
    console.log("\n=== SEARCH RESULTS ===\n");
    console.log(`Answer Count: ${retrieveDocs.length}`);

    console.log("\n===Sending this to Mistral Model===\n")
    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system", "You are an automated program repair tool which works as an expert in CSS and HTML."],
        ["user", `Fix the following responsive layout failure using the provided context: 
            RLF Type: {RLF_type},
            Type Definition: {Type_definition},
            Failure element XPaths: {Failure_element_XPaths},
            Viewport range: {viewport_range},
            localized property which is causing the failure: {localized_property},
            screenshot of the failure region: {screenshot_failure},
            screenshot of the lower and upper bound layouts: {screenshot_lower_bound}, {screenshot_upper_bound},
            5 relevant stack overflow threads containing answers and comments: {relevant_stack_overflow_threads}.
            Only return the repaired value of the localized property, do not include details. Ensure to keep the web layout responsive (DO NOT USE px, try to use rem, em, or %) and maintain the original design.
            Let's think step by step.`],
    ]);

    const chain = promptTemplate.pipe(model).pipe(new StringOutputParser());

    const response = await chain.invoke({ 
            RLF_type: "Element Collision", 
            Type_definition: "Elements collide into one another due to insufficient accommodation space when viewport width reduces.",
            Failure_element_XPaths: 'Node 1:/HTML/BODY/HEADER, Node 2: /HTML/BODY/DIV',
            viewport_range: '600-680',
            localized_property: 'margin-bottom: 20px of Node: /HTML/BODY/HEADER',
            screenshot_failure: encodeImage('FID-1-element-collision-320-680-capture-320-TP.png'),
            screenshot_lower_bound: null,
            screenshot_upper_bound: encodeImage('FID-1-element-collision-320-680-capture-681-FP.png'),
            relevant_stack_overflow_threads: JSON.stringify(retrieveDocs)
     });

     console.log(response);
}


async function create_knowledge_base(threads, collectionName) {
    const SO_collection = threadToDocument(threads, textSplitter);
    const documents = await processDocuments(SO_collection, textSplitter);
    const vectordb = await storeInVectorDB(documents, collectionName);
    return vectordb;
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

        if (thread.answers) {
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

async function storeInVectorDB(finalDocs, collectionName) {
    
    // Store in ChromaDB with batching
    console.log("Storing documents in ChromaDB...");
    
    try {
        const initDocs = finalDocs.slice(0, 1);
        let vectorStore = await Chroma.fromDocuments(
            initDocs,
            embeddings,
            { 
                collectionName: collectionName,
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


main()