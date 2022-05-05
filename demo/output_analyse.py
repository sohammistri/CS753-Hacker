import json

file_path = "../output.json"

with open(file_path) as f:
    data = json.load(f)
    transcript = data["output"][0]["transcription"]
    offset = data["output"][0]["offsets"]
    print(len(transcript), len(offset))