import { MistralAI } from "@langchain/mistralai";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import 'dotenv/config';

const llm = new MistralAI({
  apiKey: process.env.MISTRL_API_KEY || 'your_api_key',
  model: "codestral-latest",
  temperature: 0,
  maxTokens: undefined,
  maxRetries: 2,
});

try {
  const loader = new CSVLoader('submission.csv');
  const data = await loader.load();
  console.log(data[0]);

  const inputText = "MistralAI is an AI company that ";
  const completion = await llm.invoke(inputText);
  console.log(completion);
} catch (error) {
  console.error('Error:', error.message);
}