import codecs


def main():
    # 単語リスト
    words = {}
    # 単語数
    words_count = 1

    # 読込
    f = codecs.open("data\caption-wakati.txt", "r", "utf-8")
    # 1行ずつ処理を行う
    line = f.readline()
    while line:
        # 不要文字を削除し分割
        l = line.strip().split(" ")
        # 単語を数値に変換
        for x in l:
            if x not in words:
                words_count += 1
                words[x] = words_count
        # 次の行を読み込む
        line = f.readline()

    # クローズ処理
    f.close()

    # 結果の保存
    f = codecs.open("data\caption-words.txt", "w", "utf-8")
    for w in words:
        # 番号,単語の形式で書き込み
        f.write(str(words[w]) + "," + w + "\n")

    # クローズ処理
    f.close()

if __name__ == "__main__":
    main()