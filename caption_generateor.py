import json

images = {}
captions = {}

with open("data\yjcaptions26k_clean.json", "r", encoding="utf-8") as f:
    jsons = json.load(f)
    for i in jsons["annotations"]:
        # 説明
        caption = i["caption"]
        print(caption)