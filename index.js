import { Mistral } from '@mistralai/mistralai';
import 'dotenv/config';

const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';
console.log('API Key:', apiKey);

const client = new Mistral({apiKey: apiKey});

const chatResponse = await client.chat.complete({
    model: 'mistral-tiny',
    messages: [
        {
            role: 'user',
            content: 'Hello, where is niagra falls?',
        },
    ],
    temperature: 0.6
})

console.log(chatResponse.choices[0].message.content);