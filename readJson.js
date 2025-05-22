import fs from 'fs';

function readJson() {
    fs.readFile('rlf_full_list.json', 'utf8', (err, data) => {
        if (err) {
            console.error("Error reading file:", err);
            return;
        }

        try {
            const json = JSON.parse(data);
            return json
        } catch (parseError) {
            console.error("Failed to parse JSON:", parseError);
        }
    });
}

export default readJson;

