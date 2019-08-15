import codecs
import numpy as np
import os
import pickle
import sys
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.links.caffe import CaffeFunction
from PIL import Image

# バッチサイズ
batch_size = 10
# GPUを使用
uses_device = 0

# cupy使用チェック
if uses_device >= 0:
    import cupy as cp
else:
    cp = np

class ImageCaption_NN(chainer.Chain):

    def __init__(self, n_words, n_units):
        super(ImageCaption_NN, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(4096, n_units)
            self.embed = L.EmbedID(n_words, n_units)
            self.l2 = L.LSTM(n_words, n_units)
            self.l3 = L.LSTM(n_units, n_units)
            self.l4 = L.Linear(n_units, n_words)

    def encode(self, x):
        c = self.l1(x)
        return c

    def decode(self, x):
        e = self.embed(x)
        h1 = self.l2(e)
        h2 = self.l3(h1)
        c = self.l4(h2)
        return c

    def get_state(self):
        return self.l1.c, self.l1.h

    def set_state(self, c, h):
        self.l2.set_state(c, h)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

class ImageCaptionUpdater(training.StandardUpdater):

    def __init__(self, iter, optimizer, device):
        super(ImageCaptionUpdater, self).__init__(
            iter,
            optimizer,
            device=device
        )

    def update_core(self):
        # 損失
        loss = 0

        # IteratorとOptimizerを取得
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        # ニューラルネットワークを取得
        model = optimizer.target

        # 1バッチ分取得
        x = train_iter.__next__()
        # 画像ベクトルと解説文を取得
        vectors = [s[0] for s in x]
        words = [s[i] for s in x]

        # RNNのステータスをリセットする
        model.reset_state()

        # 画像ベクトルを入力
        for i in range(5):
            v = [s[i] for s in vectors]
            model.encode(cp.array(v, dtype=cp.float32))

        # RNNのステータスをリセットする
        c, h = model.get_state()
        model.set_state(c, h)

        # 文の長さだけ繰り返しRNNに学習
        for i in range(len(words[0]) - 1):
            # バッチ処理用の配列化
            batch = cp.array([s[i] for s in words], dtype=cp.int32)
            # 正解データの配列
            t = cp.array([s[i + 1] for s in words], dtype=cp.int32)
            # 全て終端文字の場合中断
            if cp.min(batch) == 1 and cp.max(batch) == 1:
                break
            # RNNを実行
            y = model.decode(batch)
            # 結果との比較
            loss += F.softmax_cross_entropy(y, t)

        # 重みをリセットする
        optimizer.target.cleargrads()
        # 誤差逆伝播
        loss.backward()
        # 新しい重みで更新
        optimizer.update()

def main():
    # 単語リスト
    words = {}

    # 単語を読み込む
    f = codecs.open("data\\caption-words.txt", "r", "utf-8")

    line = f.readline()
    while line:
        # 不要文字を削除し分割
        l = line.strip().split(",")
        words[l[1]] = int(l[0])
        # 次の行を読み込む
        line = f.readline()
    # クローズ処理
    f.close()

    # 読込
    s_w = codecs.open("data\\caption-wakati.txt", "r", "utf-8")
    s_i = codecs.open("data\\img_id.txt", "r", "utf-8")

    # 全ての画像ベクトルと説明文のセット
    sentence = []

    # TODO:CaffeModel=>pickle化
    # ReferenceError: weakly-referenced object no longer exists

    MODEL = "model\\bvlc_alexnet.caffemodel"
    # PICKLE = "model\\alex_net.pkl"
    # pickleを読込
    # if os.path.exists(PICKLE):
    #     # 存在する場合
    #     model = pickle.load(open(PICKLE, "rb"))

    # else:
    #     # 存在しない場合
    #     if os.path.exists(MODEL):
    #         CaffeModelをpickleに変換する
    #         model = CaffeFunction(MODEL)
    #         pickle.dump(model, open(PICKLE, "wb"))
    #         model = pickle.load(open(PICKLE, "rb"))
    #     else:
    #         CaffeModelが存在しない場合は中断
    #         print("model notfound")
    #         exit()

    model = CaffeFunction(MODEL)

    if uses_device >= 0:
        # GPUを使う
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()

    # 1行ずつ処理を行う
    line = s_w.readline()
    img_id = s_i.readline()

    while line and img_id:
        # 行中の単語をリスト化
        l = line.strip().split(" ")
        # ファイル名を作成
        file_name = img_id.strip() + ".jpg"
        # デバッグ
        print(file_name)
        # ファイルの読込
        img = Image.open(file_name).resize((400, 400)).convert("RGB")
        # 画像ベクトルの配列
        vectors = []
        # 4辺+中央で計5枚の画像を作る
        for s in [
                (0, 0, 227, 227),     # 左上
                (173, 0, 400, 277),   # 右上
                (0, 173, 227, 400),   # 左下
                (173, 173, 400, 400), # 右下
                (86, 86, 313, 313)    # 中央
            ]:
            # 画像から切り出し
            cropimg = img.crop(s)
            # 画素を数値データに変換
            pix = np.array(cropimg, dtype=np.float32)
            pix = (pix[::-1]).transpose(2, 0, 1)
            x = cp.array([pix], dtype=cp.float32)
            # fc6層のデータを抽出
            e, = model(inputs={"data": x}, outputs=["fc6"], disable=["drop6"])
            # 画像ベクトルの配列に結果を格納
            vectors.append(e.data[0].copy())
        # 数値の配列
        lines = [0]
        # 単語を数値に変換
        for x in l:
            if x in words:
                lines.append(words[x])
        # 行が終わったところで終端文字を挿入
        lines.append(1)
        sentence.append((vectors, lines))
        # 次の行
        line = s_w.readline()
        img_id = s_i.readline()

    # クローズ処理
    s_w.close()
    s_i.close()

    # 最長の文
    l_max = max([len(l[1]) for l in sentence])

    # 文長を揃える(バッチ処理の関係)
    for i in range(len(sentence)):
        # 足りない長さは終端文字で埋める
        sentence[i][1].extend([1] * (l_max - len(sentence[i][1])))

    # ニューラルネットワークの作成
    model = ImageCaption_NN(max(words.values()) + 1, 500)

    if uses_device >= 0:
        model.to_gpu()

    # Optimizerを作成
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Iteratorを作成
    train_iter = iterators.SerialIterator(sentence, batch_size, shuffle=False)

    updater = ImageCaptionUpdater(train_iter, optimizer, device=uses_device)
    trainer = training.Trainer(updater, (80, "epoch"), out="result")

    # 学習状況を可視化
    trainer.extend(extensions.ProgressBar(update_interval=1))
    # 学習実行
    trainer.run()

    # 結果を保存する
    chainer.serializers.save_hdf5("result.hdf5", model)

if __name__ == "__main__":
    main()