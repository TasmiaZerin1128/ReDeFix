import fs from 'fs/promises';

async function readJson() {
  try {
    const data = await fs.readFile('rlf_full_list.json', 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error("Error reading or parsing file:", error);
    return null;
  }
}

export default readJson;
