# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn
import argparse
import os
import numpy as np
import math
import sys
from collections import defaultdict
import pdb
import pickle

def create_pair(user_list):
    pair = []
    for user, item_list in user_list.items():
        for item in item_list:
            pair.extend([(user, item)])
    return pair

training_user_set,training_item_set,training_set_count = np.load('./data/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load('./data/testing_set.npy',allow_pickle=True)
user_rating_set_all = np.load('./data/user_rating_set_all.npy',allow_pickle=True).item()

user_size = 52643
item_size = 91599
train_length = np.zeros(user_size)
for user_id in training_user_set:
    train_length[user_id] = len(training_user_set[user_id])

sample_test_user_size = 100
chosen_test_indice = np.where(train_length<sample_test_user_size)[0]
num_unlearnable_users = 50
# np.random.seed(2022)
# sample50 = np.random.choice(chosen_test_indice, size=num_unlearnable_users, replace=False)
# np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-2022.npy", sample50)
# np.random.seed(5)
# sample50 = np.random.choice(chosen_test_indice, size=num_unlearnable_users, replace=False)
# np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-5.npy", sample50)
# np.random.seed(10)
# sample50 = np.random.choice(chosen_test_indice, size=num_unlearnable_users, replace=False)
# np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-10.npy", sample50)
np.random.seed(15)
sample50 = np.random.choice(chosen_test_indice, size=num_unlearnable_users, replace=False)
np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-15.npy", sample50)

train_pair = create_pair(training_user_set)
test_pair = create_pair(testing_user_set)

dataset = {'user_size': user_size, 'item_size': item_size, 'all_user_list': user_rating_set_all,
               'train_user_list': training_user_set, 'test_user_list': testing_user_set,
               'train_pair': train_pair, 'test_pair': test_pair,
               'train_item_list': training_item_set}
dirname = './preprocessed/'
filename = './preprocessed/amazon-'+str(num_unlearnable_users)+'.pickle'
os.makedirs(dirname, exist_ok=True)
with open(filename, 'wb') as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
