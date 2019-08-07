import sys
import json
import codecs
import urllib.request

# 画像リスト
images = {}
# 説明文リスト
captions = {}

with open("data\yjcaptions26k_clean.json", "r", encoding="utf-8") as f:
    jsons = json.load(f)
    for i in jsons["annotations"]:
        # 説明文
        caption = i["caption"]
        # 画像ID
        img_id = i["image_id"]
        if img_id not in images:
            # 画像のエントリを保存
            img = [j["flickr_url"] for j in jsons["images"] if j["id"] == img_id]
            if len(img) > 0:
                # 画像リストの追加
                images[img_id] = img[0]

        # 内包表記を使用しない場合
        # if img_id not in images:
        #     for j in jsons["images"]:
        #         if j["id"] == img_id:
        #             img = j["flickr_url"]

        # 説明文リストの追加
        captions[img_id] = caption

f = codecs.open("caption.txt", "w", "utf-8")
o = codecs.open("img_id.txt", "w", "utf-8")

for k in images:
    try:
        # 画像リンクアクセス
        res = urllib.request.urlopen(images[k])
    except urllib.error.HTTPError:
        # レスポンスエラーの場合は次処理へ
        continue

    print(res.status)

    # クローズ処理
    res.close()