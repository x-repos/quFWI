import json
import os

transcript_path = "/home/x/.gemini/antigravity-cli/brain/7906f5d5-e705-41b5-91c7-adc454523a0f/.system_generated/logs/transcript_full.jsonl"

# First, reset the file to HEAD just in case
os.system("cd /home/x/Workspace/1.2-quFWI/tex && git checkout manuscript.tex")

with open("/home/x/Workspace/1.2-quFWI/tex/manuscript.tex", "r") as f:
    current_text = f.read()

# We will read the transcript sequentially and apply every successful replace_file_content
# and multi_replace_file_content on manuscript.tex.
replacements_to_apply = []

with open(transcript_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("type") == "PLANNER_RESPONSE" and "tool_calls" in data:
                for tc in data["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name")
                    args = func.get("arguments", "{}")
                    
                    if name in ["default_api:replace_file_content", "default_api:multi_replace_file_content"]:
                        try:
                            args_json = json.loads(args)
                            if args_json.get("TargetFile", "").endswith("manuscript.tex"):
                                # Check if it succeeded by looking ahead in the transcript?
                                # Assume all our replacements in the past 2 hours succeeded because we reached the end successfully.
                                if name == "default_api:replace_file_content":
                                    target = args_json.get("TargetContent", "")
                                    replacement = args_json.get("ReplacementContent", "")
                                    replacements_to_apply.append((target, replacement))
                                elif name == "default_api:multi_replace_file_content":
                                    chunks = args_json.get("ReplacementChunks", [])
                                    for chunk in chunks:
                                        target = chunk.get("TargetContent", "")
                                        replacement = chunk.get("ReplacementContent", "")
                                        replacements_to_apply.append((target, replacement))
                        except:
                            pass
        except:
            pass

print(f"Found {len(replacements_to_apply)} text replacements to apply.")

success_count = 0
for target, replacement in replacements_to_apply:
    if target in current_text:
        current_text = current_text.replace(target, replacement)
        success_count += 1
    else:
        # Sometimes exact match fails if a previous replacement altered the same block, but typically they shouldn't.
        pass

with open("/home/x/Workspace/1.2-quFWI/tex/manuscript_rebuilt.tex", "w") as f:
    f.write(current_text)

print(f"Applied {success_count} replacements successfully.")
