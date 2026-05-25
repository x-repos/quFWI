import json

transcript_path = "/home/x/.gemini/antigravity-cli/brain/7906f5d5-e705-41b5-91c7-adc454523a0f/.system_generated/logs/transcript_full.jsonl"

with open(transcript_path, 'r') as f:
    for line in f:
        if "diff --git" in line:
            with open('raw_line.txt', 'w') as out:
                out.write(line)
            break
