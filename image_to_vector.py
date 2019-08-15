from chainer.links.caffe import CaffeFunction
import pickle
import os


# モデルを読込
if os.path.exists("model\\alex_net.pkl"):
    # pickleが存在する場合
    model = pickle.load(open("model\\alex_net.pkl", "rb"))

else:
    # pickleが存在しない場合
    if os.path.exists("model\\bvlc_alexnet.caffemodel"):
        # CaffeModelをpickleに変換する
        model = CaffeFunction("model\\bvlc_alexnet.caffemodel")
        pickle.dump(model, open("model\\alex_net.pkl", "wb"))
        model = pickle.load(open("model\\alex_net.pkl", "rb"))
    else:
        # CaffeModelが存在しない場合は中断
        print("model notfound")
        exit()

# print(type(model))