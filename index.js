import { Mistral } from '@mistralai/mistralai';
import 'dotenv/config';
import fs from 'fs';

const apiKey = process.env.MISTRL_API_KEY || 'your_api_key';
console.log('API Key:', apiKey);

const client = new Mistral({apiKey: apiKey});

async function encodeImage(imagePath) {
    try {
        
        const imageBuffer = fs.readFileSync(imagePath);

        // Convert the buffer to a Base64-encoded string
        const base64Image = imageBuffer.toString('base64');
        return base64Image;
    } catch (error) {
        console.error(`Error: ${error}`);
        return null;
    }
}

const imagePath = "FID-19-element-protrusion-697-768-capture-697-TP.png"
const base64Image = await encodeImage(imagePath)


const chatResponse = await client.chat.complete({
    model: 'pixtral-12b',
    messages: [
        {
            role: 'user',
            content: [
                { type: "text", text: "The red dashed area shows the element overflowing its container, which is shown in yellow dash. Can you tell me how to repair this?" },
                {
                  type: "image_url",
                  imageUrl: `data:image/jpeg;base64,${base64Image}`,
                },
              ],
        },
    ],
})

console.log(chatResponse.choices[0].message.content);