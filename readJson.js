import fs from 'fs';

function handleItem(item) {
  console.log(item);
}

function processJsonData(jsonData) {
  if (!Array.isArray(jsonData)) {
    console.error("Invalid JSON format: Expected an array.");
    return;
  }

  for (let index = 0; index < jsonData.length; index++) {
    const item = jsonData[index];
    handleItem(item);
    break;
  }
}

function main() {
    fs.readFile('rlf_full_list.json', 'utf8', (err, data) => {
        if (err) {
            console.error("Error reading file:", err);
            return;
        }

        try {
            const json = JSON.parse(data);
            processJsonData(json);
        } catch (parseError) {
            console.error("Failed to parse JSON:", parseError);
        }
    });
}

main()

