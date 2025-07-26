import { ChatMistralAI } from "@langchain/mistralai";
import 'dotenv/config';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import retrieve from "./retrieval.js";
import encodeImage from "./sendImage.js";
import readJson from "./readJson.js";
import fs from 'fs/promises';
import path from 'path';

const DEFINITIONS = {
    "Element-Collision": "Elements collide into one another due to insufficient accommodation space when viewport width reduces.",
    "Element-Protrusion": "When the child element is contained within its container, but as the viewport width decreases, it lacks sufficient space to fit within its parent. As a result, the child element protrudes out of its container.",
    "Viewport-Protrusion": "As the viewport size decreases, elements may not only overflow their containers but also protrude out of the viewable area of the webpage (i.e., the <BODY> tag), causing them to appear outside the horizontally visible portion of the page.",
    "Wrapping Elements": "When the container is not wide enough but has a flexible height, horizontally aligned elements contained within it no longer fit side by side, causing “wrap” to a new line on the page."
}

let collision_collection = "SO_collision";
let protrusion_collection = "SO_protrusion";
let wrapping_collection = "SO_wrapping";

const COLLECTIONS = {
    "Element Collision": collision_collection,
    "Element Protrusion": protrusion_collection,
    "Viewport Protrusion": protrusion_collection,
    "Wrapping Elements": wrapping_collection
}

const apiKey = process.env.MISTRAL_API_KEY || "your_api_key";
console.log("API Key:", apiKey);

const model = new ChatMistralAI({
  model: "mistral-small-2503",
  apiKey: apiKey,
  temperature: 0,
});

async function main() {
    let failureData = null;
    const jsonData = await readJson();
    if (Array.isArray(jsonData)) {
        for (const item of jsonData) {
            failureData = {
                id: item['id'],
                failureType: item['type'],
                failureRange: item['viewportRange'],
                failureNode: item['failureNode'],
                failureParent: item['failureParent'],
                failureNodeRect: item['failureElementRect'],
                failureMinScreenshot: item['screenshotFailure'],
                failureOuterUpperSS: item['screenshotUpperBound'],
                faultyProperties: item['localizedProperty']
            };
            await createPrompt(failureData);
        }
    }
}

async function createPrompt(failureData) {
    console.log("Retrieve first============\n")
    console.log(failureData.failureType);
    const propertyNames = [...new Set(failureData.faultyProperties.map(element => Object.keys(element)[0]))];
    const retrieveDocs = await retrieve(
        COLLECTIONS[failureData.failureType],
        propertyNames
    );

    console.log("\n===Sending this to Mistral Model===\n");
    const promptTemplate = ChatPromptTemplate.fromMessages([
        [
        "system",
        "You are an automated program repair tool which works as an expert in CSS and HTML.",
        ],
        [
        "user",
        `Fix the following responsive layout failure using the provided context: 
            RLF Type (means which type of failure): {RLF_type},
            Type Definition (how does the failure occur): {Type_definition},
            Failure element XPaths: {Failure_element_XPaths},
            Failure segment coordinates (region which is out of the container/colliding/going to a new line): {Failure_element_rect},
            Failure Viewport range: {viewport_range},
            Localized properties which are causing the failure (ranked from most to least problematic) [each tuple contains the property: value, then the element]: {localized_property},
            screenshot of the failure region (red and yellow dashed region): {screenshot_failure},
            screenshot of the upper bound layout where there is no failure (red and yellow dashed region): {screenshot_upper_bound},
            example stack overflow threads to help you understand how developers solve these failures. Each containing the problem, answers and comments on how to repair that failure: {relevant_stack_overflow_threads}.
            Generate a repair patch from the given localized properties, fix which are needed so that the failure is resolved. Or you can add additional properties, if their absence is causing the failure. Do not provide explanations or notes.
            Keep the layout as close to the original, as shown in upper bound screenshot.
            Let's solve this step by step.`,
        ],
    ]);

    const chain = promptTemplate.pipe(model).pipe(new StringOutputParser());

    const response = await chain.invoke({
        RLF_type: failureData.failureType,
        Type_definition: DEFINITIONS[failureData['failureType']],
        Failure_element_XPaths: `Node 1:${failureData.failureNode}, Node 2: ${failureData.failureParent ? failureData.failureParent : ""}`,
        Failure_element_rect: `Node 1:${failureData.failureNodeRect}`,
        viewport_range: failureData.failureRange,
        localized_property: failureData.faultyProperties,
        screenshot_failure: failureData.failureMinScreenshot ? encodeImage(
        failureData.failureMinScreenshot
        ) : null,
        screenshot_upper_bound: failureData.failureOuterUpperSS ? encodeImage(
        failureData.failureOuterUpperSS
        ) : null,
        relevant_stack_overflow_threads: JSON.stringify(retrieveDocs),
    });

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fileName = `repair_ID_${failureData.id}_${failureData.failureType}_${timestamp}.txt`;
    const filePath = path.join(process.cwd(), 'repairs', fileName);
    
    // Make sure the directory exists
    await fs.mkdir(path.join(process.cwd(), 'repairs'), { recursive: true });
    
    await fs.writeFile(filePath, response);
    await fs.appendFile(filePath, "\n======================\n");
    
    console.log(`Response saved to: ${filePath}`);
}

main()