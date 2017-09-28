
def auc(labels, predicts):

    out = sorted(zip(labels, predicts), key=lambda x:x[1], reverse=True)

    total_positive = sum(labels)
    total_negetive = len(labels) - total_positive

    acc = 0
    neg_remain = total_negetive
    i = 0
    while i < len(labels):
        if out[i][0] == 1:
            acc += neg_remain
        else:
            neg_remain -= 1

        i += 1

    return float(acc) / float(total_positive * total_negetive)


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import ranking
    total = 50000
    p = np.random.rand(total)
    l = np.random.randint(0, 2, total)
    print ranking.roc_auc_score(l, p)

    sub_p = np.split(p, 5)
    sub_l = np.split(l, 5)
    sub_auc = map(lambda x: ranking.roc_auc_score(sub_l[x], sub_p[x]), range(5))
    print sum(sub_auc) / 5.0, sub_auc
#    print ranking.roc_auc_score(l, p)
