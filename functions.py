import numpy as np

def softmax(x):
    # 自分の書いたコードではエラーexp部分がオーバーフロー
    # sum_exp = np.sum(np.exp(x))
    # return np.exp(x) / sum_exp
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(pred, true):
    data_size = true.shape[0]
    
    # if true.ndim == 1:
        # ラベルデータの場合の変換を追加する
        # true.eye()

    # one-hot表現として処理
    # logのときに0だとinfになるので極小値を与える。ただ1超える可能性あるけどいいの？
    out = true * (-1) * np.log(pred + 1e-7)

    return np.sum(out) / data_size
