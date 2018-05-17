#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
import pickle
import functools
import itertools
import numpy as np
from pprint import pprint
from termcolor import cprint
from datetime import datetime

from pbil.optimizer import optimize
from load_dataset import load_and_preprocess
from tensorflow_classifier2 import train_and_test

from colnames import *

#######################################################################

counter = 0
l = []
score_max = 0.0

df = load_and_preprocess('TrainingSet(3).csv')

def eval_fun(colnames, bits):
    global df, counter, l, score_max
    
    counter += 1
    cprint(counter, 'white', 'on_green')

    spl = [list(y) for x, y in itertools.groupby(zip(colnames, bits), lambda z: z[0] == None) if not x]
    colnames_categ_use = [x[0] for x in spl[0] if x[1]]
    colnames_numerical_use = [x[0] for x in spl[1] if x[1]]

    pprint(colnames_categ_use)
    print('')
    pprint(colnames_numerical_use)
    score = train_and_test(df, 1, colnames_categ_use, colnames_numerical_use)

    if os.path.exists('checkpoints.pkl'):
        with open('checkpoints.pkl', 'rb') as flo:
            data = pickle.load(flo)
    else:
        data = []

    with open('checkpoint.pkl', 'wb') as flo:
        print('Saving checkpoint')
        data.append((score, zip(colnames, bits), l, datetime.utcnow()))
        pickle.dump(data, flo, pickle.HIGHEST_PROTOCOL)
    
    cprint(score, 'red', 'on_yellow')
    return score

if __name__ == '__main__':
    # print("done!")
    # print(score)
    data = []
    data.extend(selected_feature_names_categ)
    data.extend([None])
    data.extend(selected_feature_names_interval)

    pbil_params = {
        'learn_rate': 0.05,
        'neg_learn_rate': 0.05,
        'pop_size': 15,
        'num_best_vec_to_update_from': 2,
        'num_worst_vec_to_update_from': 2,
        'vec_len': len(data),
        'optimisation_cycles': 8, # PBIL n_epochs
        'eval_f': functools.partial(eval_fun, data),
        'vec_storage': l,
    }

    result = optimize(**pbil_params)
    print(result)
    # pprint(l)

    for popvec in l:
        print(', '.join(["{:.2f}".format(item) for item in popvec]))
