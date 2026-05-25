import json
import re

with open('raw_line.txt', 'r') as f:
    line = f.read()

try:
    data = json.loads(line)
    
    # We are looking for tool_responses
    if 'tool_responses' in data:
        for tr in data['tool_responses']:
            resp = tr.get('response', {})
            output = resp.get('output', '')
            if 'diff --git' in output:
                # the diff might have the prefix "The command completed successfully.\n\t\t\t\tOutput:\n\t\t\t\t"
                if "Output:\n" in output:
                    diff_text = output.split("Output:\n")[-1].strip()
                    # if it's truncated, it might have <truncated ... lines>
                    # But the transcript might not be truncated if we are lucky!
                    with open('recover.patch', 'w') as out:
                        out.write(diff_text)
                    print("Written to recover.patch")
except Exception as e:
    print(e)
