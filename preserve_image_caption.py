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

for image in images:
    try:
        # 画像リンクアクセス
        res = urllib.request.urlopen(images[image])
    except urllib.error.HTTPError:
        # レスポンスエラーの場合は次処理へ
        continue

    # レスポンス読込
    res_body = res.read()

    # 画像保存
    jpg = codecs.open(str(image) + ".jpg", "w", "utf-8")
    jpg.buffer.write(res_body)

    # 説明文と画像IDを保存
    f.write(captions[image] + "\n")
    o.write(str(image) + "\n")

    # クローズ処理
    jpg.close()
    res.close()

# クローズ処理
f.close()
o.close()