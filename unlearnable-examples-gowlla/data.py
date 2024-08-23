import pdb
from collections import defaultdict
import numpy as np
import os
import pickle
import pandas as pd


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def preprocessing_gowalla(file_loc='./gowalla/'):
    train_path = os.path.join(file_loc, 'train.csv')
    df = pd.read_csv(train_path)
    test_path = os.path.join(file_loc, 'test.csv')
    df_test = pd.read_csv(test_path)
    print('Complete assigning unique index to user and item')
    user_size = df['uid'].max()+1
    item_size = df['sid'].max()+1
    print(user_size)
    print(item_size)
    all_user_list = [set() for u in range(user_size)]
    train_user_list = [set() for u in range(user_size)]
    test_user_list = [set() for u in range(user_size)]
    train_length = np.zeros(user_size)
    test_length = np.zeros(user_size)
    for idx, row in df.iterrows():
        # pdb.set_trace()
        user = row['uid']
        item = row['sid']
        train_user_list[user].add(item)
        all_user_list[user].add(item)
        train_length[user] += 1
    for idx, row in df_test.iterrows():
        user = row['uid']
        item = row['sid']
        test_user_list[user].add(item)
        all_user_list[user].add(item)
        test_length[user] += 1

    test_indice = np.where(train_length<14)[0]

    print(test_indice.shape)
    # pdb.set_trace()
    np.random.seed(2022)
    num_unlearnable_users = 1000
    sample50 = np.random.choice(test_indice, size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample50-2022.npy", sample50)
    np.random.seed(5)
    sample50 = np.random.choice(test_indice, size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample50-5.npy", sample50)
    np.random.seed(10)
    sample50 = np.random.choice(test_indice, size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample50-10.npy", sample50)
    np.random.seed(15)
    sample50 = np.random.choice(test_indice, size=num_unlearnable_users, replace=False)
    np.save("./preprocessed/sample50-15.npy", sample50)

    pdb.set_trace()
    train_item_list = uim2ium(train_user_list)
    train_pair = create_pair(train_user_list)
    test_pair = create_pair(test_user_list)

    dataset = {'user_size': user_size, 'item_size': item_size, 'all_user_list': all_user_list,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair, 'test_pair': test_pair,
               'train_item_list': train_item_list}
    dirname = './preprocessed/'
    filename = './preprocessed/gowalla.pickle'
    os.makedirs(dirname, exist_ok=True)
    pdb.set_trace()
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def uim2ium(user_list):
    item_list = defaultdict(set)
    for user,item_set in enumerate(user_list):
        for item in item_set:
            item_list[int(item)].add(user)
    return item_list


def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair


if __name__ == '__main__':
    preprocessing_gowalla()
