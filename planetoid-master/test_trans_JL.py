
from scipy import sparse as sp
from trans_model import trans_model as model
import argparse
import pickle
import numpy as np


def train_transductive(dataset_name, args, x, y, tx, ty, graph, max_iter=10000, init_iter_label=2000, init_iter_graph=70, tolerance=1e-4):

    def comp_accu(tpy, ty):
        return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

    x = x.astype(np.float32)
    tx = tx.astype(np.float32)
    y = y.astype(np.int32)
    ty = ty.astype(np.int32)

    m = model(argparse.Namespace(**args))                                             # initialize the model
    m.add_data(x, y, graph)                                     # add data
    m.build()                                                   # build the model
    m.init_train(init_iter_label = init_iter_label, init_iter_graph = init_iter_graph)  # pre-training
    iter_cnt, max_accu = 0, 0
    for iter in range(max_iter):
        m.step_train(max_iter = 1, iter_graph = 0, iter_inst = 1, iter_label = 0)   # perform a training step
        tpy = m.predict(tx)
        accu = comp_accu(tpy, ty)                                                  # compute the accuracy on the dev set
        print (iter_cnt, accu, max_accu)
        iter_cnt += 1
        if accu > max_accu:
            m.store_params()                                                        # store the model if better result is obtained
            max_accu = max(max_accu, accu)
        norm = np.linalg.norm(tpy - ty) / np.linalg.norm(tpy)
        if norm <= tolerance:
            print(norm)
            print("The process stopped after {0} iterations".format(iter))
            return tpy, accu
    train_acc = comp_accu(m.predict(x), y)
    print(train_acc)
    return tpy, accu
