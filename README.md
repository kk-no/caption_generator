# generate_caption

## 実装環境
- Windows10 Home 64bit
- Visual Studio 2017 (C++ BuildTool)
- Python 3.6.9 (miniconda3)
- Chainer 5.3.0
- Cuda 9.1
- Cupy 6.2.0 (cupy-cuda91 6.2.0)
- Mecab 0.996

## DataSet
- yjcaptions26k_clean.json

## 準備
- 形態素解析エンジンとしてMecabを使用
- dataディレクトリを作成し、yjcaptions26k_clean.jsonを格納
- imageディレクトリを作成
- preserve_image_caption.pyを実行

## TODO
- 訓練データ用意(画像とキャプション)
- 学習
- 新規画像の読込(部分)とキャプションの生成

## LINK
- YJCaption https://github.com/yahoojapan/YJCaptions
- CNN http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel