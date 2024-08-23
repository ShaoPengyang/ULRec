#-- coding:UTF-8 --
import os
import numpy as np
import math
import sys
import argparse
import pdb
import pickle
from collections import defaultdict
import time
import pandas as pd

data_dir = './data/ml-10M100K/'


class MovieLens1M():
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'ratings', 'time'])
        df = df[df['ratings'] > 4]
        return df


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def create_user_list(df, user_size):
    user_list = [dict() for u in range(user_size)]
    for row in df.itertuples():
        user_list[row.user][row.item] = 1
    return user_list


def split_train_test(user_list, test_size=0.2, val_size=0.1):
    train_user_list1 = [None] * len(user_list)
    test_user_list1 = [None] * len(user_list)
    val_user_list1 = [None] * len(user_list)
    all_user_list = [None] * len(user_list)
    user_num = 0
    for user, item_dict in enumerate(user_list):
        if len(item_dict) > 20:
            # Random select | item_dict.keys()=item_ids
            sampled_list = list(item_dict.keys())
            test_item1 = set(np.random.choice(sampled_list,
                                         size=int(len(item_dict)*test_size),
                                         replace=False))
            val_train1 = set(item_dict.keys()) - test_item1
            val_item1 = set(np.random.choice(list(val_train1),
                                            size=int(len(item_dict) * val_size),
                                            replace=False))
            test_user_list1[user] = test_item1
            val_user_list1[user] = val_item1
            train_user_list1[user] = set(item_dict.keys()) - test_item1 - val_item1
            all_user_list[user] = set(item_dict.keys())
            user_num += 1
    pdb.set_trace()
    return train_user_list1, test_user_list1,val_user_list1, all_user_list


def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair


def uim2ium(user_list):
    item_list=defaultdict(set)
    for user,item_set in enumerate(user_list):
        for item in item_set:
            item_list[int(item)].add(user)
    return item_list


def data():
    s = MovieLens1M(data_dir)
    df = s.load()

    xxx = df.groupby(['user'],as_index=False)['user'].agg({'cnt':'count'})
    user_interactions_number = dict()
    for idx, value in xxx.iterrows():
        user_interactions_number[value['user']] = value['cnt']
    df.insert(df.shape[1], 'cnt', 0)
    for idx, value in df.iterrows():
        df.loc[idx, 'cnt'] = user_interactions_number[value['user']]
    pdb.set_trace()
    df = df[df['cnt']>20]
    df, userid = convert_unique_idx(df, 'user')
    df, itemid = convert_unique_idx(df, 'item')
    print('Complete assigning unique index to user and item')
    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())
    print(user_size)
    print(item_size)

    num_unlearnable_users = 50
    np.random.seed(2022)
    sample50 = np.random.choice(np.arange(user_size), size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-2022.npy", sample50)
    np.random.seed(5)
    sample50 = np.random.choice(np.arange(user_size), size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-5.npy", sample50)
    np.random.seed(10)
    sample50 = np.random.choice(np.arange(user_size), size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample"+str(num_unlearnable_users)+"-10.npy", sample50)
    np.random.seed(15)
    sample50 = np.random.choice(np.arange(user_size), size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample" + str(num_unlearnable_users) + "-15.npy", sample50)

    total_user_list = create_user_list(df, user_size)
    train_user_list, test_user_list, val_user_list, all_user_list = split_train_test(total_user_list)
    train_item_list = uim2ium(train_user_list)
    train_pair = create_pair(train_user_list)
    test_pair = create_pair(test_user_list)
    val_pair = create_pair(val_user_list)
    print(user_size)
    print(item_size)
    dataset = {'user_size': user_size, 'item_size': item_size, 'all_user_list':all_user_list,
               'train_user_list': train_user_list, 'test_user_list': test_user_list, 'val_user_list': val_user_list,
               'train_pair': train_pair, 'test_pair': test_pair, 'val_pair': val_pair,
               'train_item_list': train_item_list}
    dirname = './preprocessed/'
    filename = './preprocessed/ml-10m.pickle'
    os.makedirs(dirname, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    data()