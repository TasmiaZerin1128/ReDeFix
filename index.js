import MistralClient from '@mistralai/mistralai';

const apiKey = process.env.MISTRL_API_TOKEN;
const client = new MistralClient(apiKey);

const chatResponse = await client.chat({
    model: 'mistral-tiny',
    messages: [
        {
            role: 'user',
            content: 'Hello, how are you?',
        },
    ],
    temperature: 0.6
})

console.log(chatResponse.choices[0].message.content);