import { MistralAI } from "@langchain/mistralai";
import 'dotenv/config';

const llm = new MistralAI({
  apiKey: process.env.MISTRL_API_KEY || 'your_api_key',
  model: "codestral-latest",
  temperature: 0,
  maxTokens: undefined,
  maxRetries: 2,
});

const inputText = "MistralAI is an AI company that ";

const completion = await llm.invoke(inputText);
console.log(completion);