
import numpy as np
from config import Config
from PIL import Image


def read_one_image(filename):
    # 画像を読み込んでモノクロ化（convert）→リサイズ
    img = np.array(Image.open(filename).convert('L').resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.ANTIALIAS))
    # (channel, height, width）の形式に変換
    img = img[np.newaxis, :]
    # 0-1のfloat値にする
    img / 255.0
    return img


def read_data(filename):
    # ファイルを開く
    f = open(filename, 'r')
    # データを入れる配列
    images = []
    labels = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # イメージを読み込み
        images.append(read_one_image(l[0]))
        # 対応するラベルを用意
        labels.append(int(l[1]))
    f.close()
    return images, labels
