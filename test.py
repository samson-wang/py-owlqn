import numpy as np
import lr
import sys

def read_breast_cancer(lr):
    for line in sys.stdin:
        try:
            i = map(int, line.strip().split(','))
        except:
            print >> sys.stderr, 'wrong instance', line.strip()
            continue
        i[0] = 10
        i[-1] = i[-1] / 4
        lr.add_instance(np.array(i[:-1]) / 10., i[-1])

def read_ajk(lr):
    for line in sys.stdin:
        i = map(int, line.strip().split())
        if i[0] > 1:
            i[0] = 1
        elif i[0] == 0:
            i[0] = 0
        lr.add_instance(np.array(i[1:]) / 1., i[0])

def test_lr():

    lr_trainer = lr.logistic_regression()
    read_ajk(lr_trainer)
    lr_trainer.train()
    lr_trainer.evaluate()

def pred_lr():

    lr_trainer = lr.logistic_regression()
    w = []
    with open(sys.argv[2]) as f:
        for k, line in enumerate(f.readlines()):
            if line.startswith('%%MatrixMarket matrix array'):
                pass
            elif line.startswith('%'):
                pass
            elif k == 1:
                row, col = map(int, line.strip().split())
                print row, col
            else:
                w.append(float(line.strip()))
    print w
    read_ajk(lr_trainer)
    lr_trainer.w = w
    lr_trainer.evaluate()

if __name__ == '__main__':

    globals()[sys.argv[1]]()
