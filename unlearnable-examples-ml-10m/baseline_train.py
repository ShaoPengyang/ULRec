import os
import sys
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from shutil import copyfile
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.autograd as autograd
import argparse
import pdb
from scipy.sparse import csr_matrix
import pickle
from collections import defaultdict
import time
import copy
from tqdm import tqdm
from data_loader import *
from evaluate import *
from CF_model import LRGCCF, BPR, WBPR, WeightedMF
from copy import deepcopy
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="change gpuid")
parser.add_argument("-g", "--gpu-id", help="choose which gpu to use", type=str, default=str(0))
parser.add_argument("-t", "--times", help="experiment id", type=str, default='0')
parser.add_argument("-c", "--choices", help="experiment id", type=int, default=0)
parser.add_argument("--dataset", help="dataset_name", type=str, default='gowalla')
parser.add_argument("--users", help="user num", type=int, default=13149)
parser.add_argument("--items", help="item num", type=int, default=14007)
parser.add_argument("--factors", help="factor num", type=int, default=64)
parser.add_argument("--batch-size", help="batch size", type=int, default=8192*4)
parser.add_argument("--lamada", help="lamada", type=float, default=0.01)
parser.add_argument("--fake-item", help="How many items are injected for a user", type=int, default=20)
parser.add_argument("--B-num", help="number of B", type=int, default=50)
parser.add_argument("--negative-item", help="How many negative items", type=int, default=5)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("-k","--sign", type=int, default=2)
args = parser.parse_args()
user_num = args.users
item_num = args.items
factor_num = args.factors
batch_size = args.batch_size
choice = args.choices
lamada = args.lamada
run_id = 't' + args.times
dataset_name = args.dataset
num_fake_item = args.fake_item
B_num = args.B_num
ng_num = args.negative_item
learning_rate = args.lr
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print("GPU id is {}, Dataset is {}, id is {}, lr is {}, batchsize is {}, fake_item num is {}, negative is {}.".format(
    str(args.gpu_id), dataset_name, run_id, str(learning_rate), str(batch_size), str(num_fake_item), str(ng_num)))
path_save_model_base = './model/' + dataset_name + '/' + run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)
path_save_log_base = './Logs/' + dataset_name + '/' + run_id
if (os.path.exists(path_save_log_base)):
    print('has log save path')
else:
    os.makedirs(path_save_log_base)
'''
-k attack user num
-c evaluate model
'''
sign = args.sign

if sign == 1:
    attacked_user_size = 50
elif sign == 2:
    attacked_user_size = 100
elif sign == 3:
    attacked_user_size = 500
else:
    attacked_user_size = 1000
print("the number of attacked_user: "+str(attacked_user_size))
Path50 = "./preprocessed/sample50-2022.npy"

def train_attack():
    print('----------------training processing------------------')
    # Load preprocess data
    with open('./preprocessed/gowalla-1000.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list = dataset['train_user_list']
        train_item_list = dataset['train_item_list']
    train_pair = []
    for user in range(user_size):
        for item in train_user_list[user]:
            train_pair.append([user, item])
    train_pair = np.array(train_pair)
    print(user_size)
    sub_items = None
    np.random.seed(0)
    if attacked_user_size == 50:
        attacked_users = np.load(Path50, allow_pickle=True)
    elif attacked_user_size == 100:
        attacked_users = np.load("./preprocessed/sample100.npy", allow_pickle=True)
    elif attacked_user_size == 500:
        attacked_users = np.load("./preprocessed/sample500.npy", allow_pickle=True)
    elif attacked_user_size == 1000:
        attacked_users = np.load("./preprocessed/sample1000-10.npy", allow_pickle=True)
    else:
        print("Please enter attacked user size number: 50/100/500/1000")
        import sys
        sys.exit()
    attacked_users_ = list(attacked_users)

    np.random.seed(2022)
    predictions_cpu = np.random.rand(len(attacked_users_), item_size)
    added_pair, fake_train_user_list, train_set_len = project_tensor(predictions_cpu, attacked_users_,
                                                                     train_user_list, item_size, user_size, sub_items)
    fake_train_pair = np.concatenate((train_pair, added_pair), axis=0)
    # pdb.set_trace()
    save_path = "./preprocessed/RND_" + str(sign) + ".npy"
    print(save_path)
    np.save(save_path, fake_train_pair)

max_epoch = 40
def evaluate(PATH):
    start_time = time.time()
    print('----------------evaluate processing------------------')
    # Load preprocess data
    with open('./preprocessed/gowalla-1000.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']
    print('-------------finish loading data---------------------')
    fake_train_pair = np.load(PATH, allow_pickle=True)
    print(len(train_pair))
    print(fake_train_pair.shape)
    new_fake_train_user_list = defaultdict(set)
    added_items = defaultdict(set)
    for _ in range(fake_train_pair.shape[0]):
        user = int(fake_train_pair[_][0])
        item = int(fake_train_pair[_][1])
        new_fake_train_user_list[user].add(item)
        if item not in train_user_list[user]:
            added_items[user].add(item)

    with open('com_add.txt','w+') as f:
        for user in range(user_size):
            f.write("user "+str(user) +': '+ str(len(added_items[user]))+" "+str(added_items[user])+'\n')

    train_set_len = 0
    for u_i in range(user_size):
        train_set_len += len(new_fake_train_user_list[u_i])
    print("len of newly fake training set is:" + str(train_set_len))
    train_dataset = TripletUniformPair(item_size, new_fake_train_user_list, ng_num, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = BPR(user_size, item_size, factor_num).cuda()
    optimizer_rec = torch.optim.Adam(model.parameters(), lr=0.005)
    print("time for prepare:" + str(time.time()-start_time))
    for epoch in range(max_epoch):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()
        train_loss_sum = []
        for idx, (u, i, j) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            j = j.cuda()
            model.zero_grad()
            loss = model(u, i, j)
            loss.backward()
            optimizer_rec.step()
            train_loss_sum.append(loss.item())
        model.eval()
        with torch.no_grad():
            user_e, item_e = model.get_embedding()
        user_e = user_e.cpu().detach().numpy()
        item_e = item_e.cpu().detach().numpy()
        all_pre = np.matmul(user_e, item_e.T)
        HR_2, NDCG_2 = [], []
        HR_4, NDCG_4 = [], []
        HR_6, NDCG_6 = [], []
        HR_8, NDCG_8 = [], []
        HR_10, NDCG_10 = [], []
        HR_20, NDCG_20 = [], []
        HR_30, NDCG_30 = [], []
        HR_40, NDCG_40 = [], []
        HR_50, NDCG_50 = [], []
        set_all = set(range(item_size))
        # all test set length for attacked users
        length = 0
        np.random.seed(0)
        # attacked_users = np.random.choice(np.arange(user_size), size=attacked_user_size, replace=False)
        if attacked_user_size == 50:
            attacked_users = np.load(Path50, allow_pickle=True)
        elif attacked_user_size == 100:
            attacked_users = np.load("./preprocessed/sample100.npy", allow_pickle=True)
        elif attacked_user_size == 500:
            attacked_users = np.load("./preprocessed/sample500.npy", allow_pickle=True)
        elif attacked_user_size == 1000:
            attacked_users = np.load("./preprocessed/sample1000-10.npy", allow_pickle=True)
        else:
            print("Please enter attacked user size number: 50/100/500/1000")
            import sys
            sys.exit()
        attacked_users = list(attacked_users)
        for u_i in attacked_users:
            fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
            item_i_list = list(fake_test_user_list)
            index_end_i = len(item_i_list)
            length += index_end_i
            if index_end_i != 0:
                item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
                item_i_list.extend(item_j_list)
                pre_one = all_pre[u_i][item_i_list]
                # indices2 = largest_indices(pre_one, 2)
                # indices4 = largest_indices(pre_one, 4)
                # indices6 = largest_indices(pre_one, 6)
                # indices8 = largest_indices(pre_one, 8)
                # indices10 = largest_indices(pre_one, 10)
                # indices20 = largest_indices(pre_one, 20)
                # indices30 = largest_indices(pre_one, 30)
                # indices40 = largest_indices(pre_one, 40)
                indices50 = largest_indices(pre_one, 50)
                indices2 = list(indices50[0][:2])
                indices4 = list(indices50[0][:4])
                indices6 = list(indices50[0][:6])
                indices8 = list(indices50[0][:8])
                indices10 = list(indices50[0][:10])
                indices20 = list(indices50[0][:20])
                indices30 = list(indices50[0][:30])
                indices40 = list(indices50[0][:40])
                indices50 = list(indices50[0])
                hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
                hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
                hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
                hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
                hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
                hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
                hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
                hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
                hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
                HR_2.append(hr_t2)
                NDCG_2.append(ndcg_t2)
                HR_4.append(hr_t4)
                NDCG_4.append(ndcg_t4)
                HR_6.append(hr_t6)
                NDCG_6.append(ndcg_t6)
                HR_8.append(hr_t8)
                NDCG_8.append(ndcg_t8)
                HR_10.append(hr_t10)
                NDCG_10.append(ndcg_t10)
                HR_20.append(hr_t20)
                NDCG_20.append(ndcg_t20)
                HR_30.append(hr_t30)
                NDCG_30.append(ndcg_t30)
                HR_40.append(hr_t40)
                NDCG_40.append(ndcg_t40)
                HR_50.append(hr_t50)
                NDCG_50.append(ndcg_t50)
        hr_test2 = round(np.mean(HR_2), 4)
        ndcg_test2 = round(np.mean(NDCG_2), 4)
        hr_test4 = round(np.mean(HR_4), 4)
        ndcg_test4 = round(np.mean(NDCG_4), 4)
        hr_test6 = round(np.mean(HR_6), 4)
        ndcg_test6 = round(np.mean(NDCG_6), 4)
        hr_test8 = round(np.mean(HR_8), 4)
        ndcg_test8 = round(np.mean(NDCG_8), 4)
        hr_test10 = round(np.mean(HR_10), 4)
        ndcg_test10 = round(np.mean(NDCG_10), 4)
        hr_test20 = round(np.mean(HR_20), 4)
        ndcg_test20 = round(np.mean(NDCG_20), 4)
        hr_test30 = round(np.mean(HR_30), 4)
        ndcg_test30 = round(np.mean(NDCG_30), 4)
        hr_test40 = round(np.mean(HR_40), 4)
        ndcg_test40 = round(np.mean(NDCG_40), 4)
        hr_test50 = round(np.mean(HR_50), 4)
        ndcg_test50 = round(np.mean(NDCG_50), 4)
        elapsed_time = time.time() - start_time
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        print('train with cluster noise')
        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
            train_loss) + ' len:' + str(length)
        print(str_print_train)
        str_print_evl2 = "attacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
        str_print_evl4 = "attacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
        str_print_evl6 = "attacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
        str_print_evl8 = "attacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
        str_print_evl10 = "attacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
        str_print_evl20 = "attacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
        str_print_evl30 = "attacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
        str_print_evl40 = "attacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
        str_print_evl50 = "attacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
        print(str_print_evl2)
        print(str_print_evl4)
        print(str_print_evl6)
        print(str_print_evl8)
        print(str_print_evl10)
        print(str_print_evl20)
        print(str_print_evl30)
        print(str_print_evl40)
        print(str_print_evl50)

        HR_2, NDCG_2 = [], []
        HR_4, NDCG_4 = [], []
        HR_6, NDCG_6 = [], []
        HR_8, NDCG_8 = [], []
        HR_10, NDCG_10 = [], []
        HR_20, NDCG_20 = [], []
        HR_30, NDCG_30 = [], []
        HR_40, NDCG_40 = [], []
        HR_50, NDCG_50 = [], []
        set_all = set(range(item_size))
        # all test set length for attacked users
        length = 0
        unattacked_users = set(range(user_size)) - set(attacked_users)
        for u_i in unattacked_users:
            fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
            item_i_list = list(fake_test_user_list)
            index_end_i = len(item_i_list)
            length += index_end_i
            if index_end_i != 0:
                item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
                item_i_list.extend(item_j_list)
                pre_one = all_pre[u_i][item_i_list]
                # indices2 = largest_indices(pre_one, 2)
                # indices4 = largest_indices(pre_one, 4)
                # indices6 = largest_indices(pre_one, 6)
                # indices8 = largest_indices(pre_one, 8)
                # indices10 = largest_indices(pre_one, 10)
                # indices20 = largest_indices(pre_one, 20)
                # indices30 = largest_indices(pre_one, 30)
                # indices40 = largest_indices(pre_one, 40)
                indices50 = largest_indices(pre_one, 50)
                indices2 = list(indices50[0][:2])
                indices4 = list(indices50[0][:4])
                indices6 = list(indices50[0][:6])
                indices8 = list(indices50[0][:8])
                indices10 = list(indices50[0][:10])
                indices20 = list(indices50[0][:20])
                indices30 = list(indices50[0][:30])
                indices40 = list(indices50[0][:40])
                indices50 = list(indices50[0])
                hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
                hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
                hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
                hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
                hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
                hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
                hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
                hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
                hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
                HR_2.append(hr_t2)
                NDCG_2.append(ndcg_t2)
                HR_4.append(hr_t4)
                NDCG_4.append(ndcg_t4)
                HR_6.append(hr_t6)
                NDCG_6.append(ndcg_t6)
                HR_8.append(hr_t8)
                NDCG_8.append(ndcg_t8)
                HR_10.append(hr_t10)
                NDCG_10.append(ndcg_t10)
                HR_20.append(hr_t20)
                NDCG_20.append(ndcg_t20)
                HR_30.append(hr_t30)
                NDCG_30.append(ndcg_t30)
                HR_40.append(hr_t40)
                NDCG_40.append(ndcg_t40)
                HR_50.append(hr_t50)
                NDCG_50.append(ndcg_t50)
        hr_test2 = round(np.mean(HR_2), 4)
        ndcg_test2 = round(np.mean(NDCG_2), 4)
        hr_test4 = round(np.mean(HR_4), 4)
        ndcg_test4 = round(np.mean(NDCG_4), 4)
        hr_test6 = round(np.mean(HR_6), 4)
        ndcg_test6 = round(np.mean(NDCG_6), 4)
        hr_test8 = round(np.mean(HR_8), 4)
        ndcg_test8 = round(np.mean(NDCG_8), 4)
        hr_test10 = round(np.mean(HR_10), 4)
        ndcg_test10 = round(np.mean(NDCG_10), 4)
        hr_test20 = round(np.mean(HR_20), 4)
        ndcg_test20 = round(np.mean(NDCG_20), 4)
        hr_test30 = round(np.mean(HR_30), 4)
        ndcg_test30 = round(np.mean(NDCG_30), 4)
        hr_test40 = round(np.mean(HR_40), 4)
        ndcg_test40 = round(np.mean(NDCG_40), 4)
        hr_test50 = round(np.mean(HR_50), 4)
        ndcg_test50 = round(np.mean(NDCG_50), 4)
        elapsed_time = time.time() - start_time
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        print('train with cluster noise')
        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
            train_loss) + ' len:' + str(length)
        str_print_evl2 = "unattacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
        str_print_evl4 = "unattacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
        str_print_evl6 = "unattacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
        str_print_evl8 = "unattacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
        str_print_evl10 = "unattacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
        str_print_evl20 = "unattacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
        str_print_evl30 = "unattacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
        str_print_evl40 = "unattacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
        str_print_evl50 = "unattacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
        print(str_print_evl2)
        print(str_print_evl4)
        print(str_print_evl6)
        print(str_print_evl8)
        print(str_print_evl10)
        print(str_print_evl20)
        print(str_print_evl30)
        print(str_print_evl40)
        print(str_print_evl50)
        print("==================================================================")


def evaluate_gcn(PATH):
    start_time = time.time()
    print('----------------evaluate processing------------------')
    # Load preprocess data
    with open('./preprocessed/gowalla-1000.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']
    print('-------------finish loading data---------------------')
    fake_train_pair = np.load(PATH, allow_pickle=True)
    # pdb.set_trace()
    new_fake_train_item_list = defaultdict(set)
    new_fake_train_user_list = defaultdict(set)
    for _ in range(fake_train_pair.shape[0]):
        user = int(fake_train_pair[_][0])
        item = int(fake_train_pair[_][1])
        new_fake_train_user_list[user].add(item)
        new_fake_train_item_list[item].add(user)

    u_d = readD(new_fake_train_user_list, user_size)
    i_d = readD(new_fake_train_item_list, item_size)
    d_i_train = u_d
    d_j_train = i_d
    sparse_u_i = readTrainSparseMatrix(new_fake_train_user_list, True, u_d, i_d, user_size, item_size)
    sparse_i_u = readTrainSparseMatrix(new_fake_train_item_list, False, u_d, i_d, user_size, item_size)

    train_set_len = 0
    for u_i in range(user_size):
        train_set_len += len(new_fake_train_user_list[u_i])
    print("len of newly fake training set is:" + str(train_set_len))
    train_dataset = TripletUniformPair(item_size, new_fake_train_user_list, ng_num, train_set_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LRGCCF(user_size, item_size, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train).cuda()
    optimizer_rec = torch.optim.Adam(model.parameters(), lr=0.001)
    print("time for prepare:" + str(time.time()-start_time))
    for epoch in range(max_epoch):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()
        train_loss_sum = []
        for idx, (u, i, j) in enumerate(train_loader):
            u = u.cuda()
            i = i.cuda()
            j = j.cuda()
            model.zero_grad()
            loss = model(u, i, j)
            loss.backward()
            optimizer_rec.step()
            train_loss_sum.append(loss.item())
        model.eval()
        with torch.no_grad():
            user_e, item_e = model.get_embedding()
        user_e = user_e.cpu().detach().numpy()
        item_e = item_e.cpu().detach().numpy()
        all_pre = np.matmul(user_e, item_e.T)
        HR_2, NDCG_2 = [], []
        HR_4, NDCG_4 = [], []
        HR_6, NDCG_6 = [], []
        HR_8, NDCG_8 = [], []
        HR_10, NDCG_10 = [], []
        HR_20, NDCG_20 = [], []
        HR_30, NDCG_30 = [], []
        HR_40, NDCG_40 = [], []
        HR_50, NDCG_50 = [], []
        set_all = set(range(item_size))
        # all test set length for attacked users
        length = 0
        np.random.seed(0)
        if attacked_user_size == 50:
            attacked_users = np.load(Path50, allow_pickle=True)
        elif attacked_user_size == 100:
            attacked_users = np.load("./preprocessed/sample100.npy", allow_pickle=True)
        elif attacked_user_size == 500:
            attacked_users = np.load("./preprocessed/sample500.npy", allow_pickle=True)
        elif attacked_user_size == 1000:
            attacked_users = np.load("./preprocessed/sample1000-10.npy", allow_pickle=True)
        else:
            print("Please enter attacked user size number: 50/100/500/1000")
            import sys
            sys.exit()
        attacked_users = list(attacked_users)
        for u_i in attacked_users:
            fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
            item_i_list = list(fake_test_user_list)
            index_end_i = len(item_i_list)
            length += index_end_i
            if index_end_i != 0:
                item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
                item_i_list.extend(item_j_list)
                pre_one = all_pre[u_i][item_i_list]
                # indices2 = largest_indices(pre_one, 2)
                # indices4 = largest_indices(pre_one, 4)
                # indices6 = largest_indices(pre_one, 6)
                # indices8 = largest_indices(pre_one, 8)
                # indices10 = largest_indices(pre_one, 10)
                # indices20 = largest_indices(pre_one, 20)
                # indices30 = largest_indices(pre_one, 30)
                # indices40 = largest_indices(pre_one, 40)
                indices50 = largest_indices(pre_one, 50)
                indices2 = list(indices50[0][:2])
                indices4 = list(indices50[0][:4])
                indices6 = list(indices50[0][:6])
                indices8 = list(indices50[0][:8])
                indices10 = list(indices50[0][:10])
                indices20 = list(indices50[0][:20])
                indices30 = list(indices50[0][:30])
                indices40 = list(indices50[0][:40])
                indices50 = list(indices50[0])
                hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
                hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
                hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
                hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
                hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
                hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
                hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
                hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
                hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
                HR_2.append(hr_t2)
                NDCG_2.append(ndcg_t2)
                HR_4.append(hr_t4)
                NDCG_4.append(ndcg_t4)
                HR_6.append(hr_t6)
                NDCG_6.append(ndcg_t6)
                HR_8.append(hr_t8)
                NDCG_8.append(ndcg_t8)
                HR_10.append(hr_t10)
                NDCG_10.append(ndcg_t10)
                HR_20.append(hr_t20)
                NDCG_20.append(ndcg_t20)
                HR_30.append(hr_t30)
                NDCG_30.append(ndcg_t30)
                HR_40.append(hr_t40)
                NDCG_40.append(ndcg_t40)
                HR_50.append(hr_t50)
                NDCG_50.append(ndcg_t50)
        hr_test2 = round(np.mean(HR_2), 4)
        ndcg_test2 = round(np.mean(NDCG_2), 4)
        hr_test4 = round(np.mean(HR_4), 4)
        ndcg_test4 = round(np.mean(NDCG_4), 4)
        hr_test6 = round(np.mean(HR_6), 4)
        ndcg_test6 = round(np.mean(NDCG_6), 4)
        hr_test8 = round(np.mean(HR_8), 4)
        ndcg_test8 = round(np.mean(NDCG_8), 4)
        hr_test10 = round(np.mean(HR_10), 4)
        ndcg_test10 = round(np.mean(NDCG_10), 4)
        hr_test20 = round(np.mean(HR_20), 4)
        ndcg_test20 = round(np.mean(NDCG_20), 4)
        hr_test30 = round(np.mean(HR_30), 4)
        ndcg_test30 = round(np.mean(NDCG_30), 4)
        hr_test40 = round(np.mean(HR_40), 4)
        ndcg_test40 = round(np.mean(NDCG_40), 4)
        hr_test50 = round(np.mean(HR_50), 4)
        ndcg_test50 = round(np.mean(NDCG_50), 4)
        elapsed_time = time.time() - start_time
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        print('train with cluster noise')
        str_print_train = "epoch:" + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
            train_loss) + ' len:' + str(length)
        print(str_print_train)
        str_print_evl2 = "attacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
        str_print_evl4 = "attacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
        str_print_evl6 = "attacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
        str_print_evl8 = "attacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
        str_print_evl10 = "attacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
        str_print_evl20 = "attacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
        str_print_evl30 = "attacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
        str_print_evl40 = "attacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
        str_print_evl50 = "attacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
        print(str_print_evl2)
        print(str_print_evl4)
        print(str_print_evl6)
        print(str_print_evl8)
        print(str_print_evl10)
        print(str_print_evl20)
        print(str_print_evl30)
        print(str_print_evl40)
        print(str_print_evl50)

        HR_2, NDCG_2 = [], []
        HR_4, NDCG_4 = [], []
        HR_6, NDCG_6 = [], []
        HR_8, NDCG_8 = [], []
        HR_10, NDCG_10 = [], []
        HR_20, NDCG_20 = [], []
        HR_30, NDCG_30 = [], []
        HR_40, NDCG_40 = [], []
        HR_50, NDCG_50 = [], []
        set_all = set(range(item_size))
        # all test set length for attacked users
        length = 0
        unattacked_users = set(range(user_size)) - set(attacked_users)
        for u_i in unattacked_users:
            fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
            item_i_list = list(fake_test_user_list)
            index_end_i = len(item_i_list)
            length += index_end_i
            if index_end_i != 0:
                item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
                item_i_list.extend(item_j_list)
                pre_one = all_pre[u_i][item_i_list]
                # indices2 = largest_indices(pre_one, 2)
                # indices4 = largest_indices(pre_one, 4)
                # indices6 = largest_indices(pre_one, 6)
                # indices8 = largest_indices(pre_one, 8)
                # indices10 = largest_indices(pre_one, 10)
                # indices20 = largest_indices(pre_one, 20)
                # indices30 = largest_indices(pre_one, 30)
                # indices40 = largest_indices(pre_one, 40)
                indices50 = largest_indices(pre_one, 50)
                indices2 = list(indices50[0][:2])
                indices4 = list(indices50[0][:4])
                indices6 = list(indices50[0][:6])
                indices8 = list(indices50[0][:8])
                indices10 = list(indices50[0][:10])
                indices20 = list(indices50[0][:20])
                indices30 = list(indices50[0][:30])
                indices40 = list(indices50[0][:40])
                indices50 = list(indices50[0])
                hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
                hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
                hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
                hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
                hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
                hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
                hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
                hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
                hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
                HR_2.append(hr_t2)
                NDCG_2.append(ndcg_t2)
                HR_4.append(hr_t4)
                NDCG_4.append(ndcg_t4)
                HR_6.append(hr_t6)
                NDCG_6.append(ndcg_t6)
                HR_8.append(hr_t8)
                NDCG_8.append(ndcg_t8)
                HR_10.append(hr_t10)
                NDCG_10.append(ndcg_t10)
                HR_20.append(hr_t20)
                NDCG_20.append(ndcg_t20)
                HR_30.append(hr_t30)
                NDCG_30.append(ndcg_t30)
                HR_40.append(hr_t40)
                NDCG_40.append(ndcg_t40)
                HR_50.append(hr_t50)
                NDCG_50.append(ndcg_t50)
        hr_test2 = round(np.mean(HR_2), 4)
        ndcg_test2 = round(np.mean(NDCG_2), 4)
        hr_test4 = round(np.mean(HR_4), 4)
        ndcg_test4 = round(np.mean(NDCG_4), 4)
        hr_test6 = round(np.mean(HR_6), 4)
        ndcg_test6 = round(np.mean(NDCG_6), 4)
        hr_test8 = round(np.mean(HR_8), 4)
        ndcg_test8 = round(np.mean(NDCG_8), 4)
        hr_test10 = round(np.mean(HR_10), 4)
        ndcg_test10 = round(np.mean(NDCG_10), 4)
        hr_test20 = round(np.mean(HR_20), 4)
        ndcg_test20 = round(np.mean(NDCG_20), 4)
        hr_test30 = round(np.mean(HR_30), 4)
        ndcg_test30 = round(np.mean(NDCG_30), 4)
        hr_test40 = round(np.mean(HR_40), 4)
        ndcg_test40 = round(np.mean(NDCG_40), 4)
        hr_test50 = round(np.mean(HR_50), 4)
        ndcg_test50 = round(np.mean(NDCG_50), 4)
        elapsed_time = time.time() - start_time
        train_loss = round(np.mean(train_loss_sum[:-1]), 4)
        print('train with cluster noise')
        str_print_evl2 = "unattacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
        str_print_evl4 = "unattacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
        str_print_evl6 = "unattacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
        str_print_evl8 = "unattacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
        str_print_evl10 = "unattacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
        str_print_evl20 = "unattacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
        str_print_evl30 = "unattacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
        str_print_evl40 = "unattacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
        str_print_evl50 = "unattacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
        print(str_print_evl2)
        print(str_print_evl4)
        print(str_print_evl6)
        print(str_print_evl8)
        print(str_print_evl10)
        print(str_print_evl20)
        print(str_print_evl30)
        print(str_print_evl40)
        print(str_print_evl50)
        print("==================================================================")


def evaluate_itemcf(PATH):
    start_time = time.time()
    print('----------------evaluate processing------------------')
    # Load preprocess data
    with open('./preprocessed/gowalla-1000.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
        train_pair = dataset['train_pair']
    print('-------------finish loading data---------------------')
    fake_train_pair = np.load(PATH, allow_pickle=True)
    # # pdb.set_trace()
    # print(len(train_pair))
    # print(fake_train_pair.shape)

    # fake_train_pair = np.array(train_pair)
    value = np.ones(fake_train_pair.shape[0])
    fake_data = csr_matrix((value, (fake_train_pair[:, 0], fake_train_pair[:, 1])), dtype=np.float64,
                           shape=(user_size, item_size))
    lts_data_tensor = sparse2tensor(fake_data).cuda()
    item_similarities = sim_matrix(lts_data_tensor.t(), lts_data_tensor.t())
    item_similarities = item_similarities.cpu().numpy()
    lts_data_tensor = lts_data_tensor.cpu().numpy()
    new_fake_train_user_list = defaultdict(set)
    added_items = defaultdict(set)
    for _ in range(fake_train_pair.shape[0]):
        user = int(fake_train_pair[_][0])
        item = int(fake_train_pair[_][1])
        new_fake_train_user_list[user].add(item)
        if item not in train_user_list[user]:
            added_items[user].add(item)

    all_pre = np.matmul(lts_data_tensor, item_similarities)
    HR_2, NDCG_2 = [], []
    HR_4, NDCG_4 = [], []
    HR_6, NDCG_6 = [], []
    HR_8, NDCG_8 = [], []
    HR_10, NDCG_10 = [], []
    HR_20, NDCG_20 = [], []
    HR_30, NDCG_30 = [], []
    HR_40, NDCG_40 = [], []
    HR_50, NDCG_50 = [], []
    set_all = set(range(item_size))
    # all test set length for attacked users
    length = 0
    np.random.seed(0)
    if attacked_user_size == 50:
        attacked_users = np.load(Path50, allow_pickle=True)
    elif attacked_user_size == 100:
        attacked_users = np.load("./preprocessed/sample100.npy", allow_pickle=True)
    elif attacked_user_size == 500:
        attacked_users = np.load("./preprocessed/sample500.npy", allow_pickle=True)
    elif attacked_user_size == 1000:
        attacked_users = np.load("./preprocessed/sample1000-10.npy", allow_pickle=True)
    else:
        print("Please enter attacked user size number: 50/100/500/1000")
        import sys
        sys.exit()
    attacked_users = list(attacked_users)
    for u_i in attacked_users:
        fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
        item_i_list = list(fake_test_user_list)
        index_end_i = len(item_i_list)
        length += index_end_i
        if index_end_i != 0:
            item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
            item_i_list.extend(item_j_list)
            pre_one = all_pre[u_i][item_i_list]
            # indices2 = largest_indices(pre_one, 2)
            # indices4 = largest_indices(pre_one, 4)
            # indices6 = largest_indices(pre_one, 6)
            # indices8 = largest_indices(pre_one, 8)
            # indices10 = largest_indices(pre_one, 10)
            # indices20 = largest_indices(pre_one, 20)
            # indices30 = largest_indices(pre_one, 30)
            # indices40 = largest_indices(pre_one, 40)
            indices50 = largest_indices(pre_one, 50)
            indices2 = list(indices50[0][:2])
            indices4 = list(indices50[0][:4])
            indices6 = list(indices50[0][:6])
            indices8 = list(indices50[0][:8])
            indices10 = list(indices50[0][:10])
            indices20 = list(indices50[0][:20])
            indices30 = list(indices50[0][:30])
            indices40 = list(indices50[0][:40])
            indices50 = list(indices50[0])
            hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
            hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
            hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
            hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
            hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
            hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
            hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
            hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
            hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
            HR_2.append(hr_t2)
            NDCG_2.append(ndcg_t2)
            HR_4.append(hr_t4)
            NDCG_4.append(ndcg_t4)
            HR_6.append(hr_t6)
            NDCG_6.append(ndcg_t6)
            HR_8.append(hr_t8)
            NDCG_8.append(ndcg_t8)
            HR_10.append(hr_t10)
            NDCG_10.append(ndcg_t10)
            HR_20.append(hr_t20)
            NDCG_20.append(ndcg_t20)
            HR_30.append(hr_t30)
            NDCG_30.append(ndcg_t30)
            HR_40.append(hr_t40)
            NDCG_40.append(ndcg_t40)
            HR_50.append(hr_t50)
            NDCG_50.append(ndcg_t50)
    hr_test2 = round(np.mean(HR_2), 4)
    ndcg_test2 = round(np.mean(NDCG_2), 4)
    hr_test4 = round(np.mean(HR_4), 4)
    ndcg_test4 = round(np.mean(NDCG_4), 4)
    hr_test6 = round(np.mean(HR_6), 4)
    ndcg_test6 = round(np.mean(NDCG_6), 4)
    hr_test8 = round(np.mean(HR_8), 4)
    ndcg_test8 = round(np.mean(NDCG_8), 4)
    hr_test10 = round(np.mean(HR_10), 4)
    ndcg_test10 = round(np.mean(NDCG_10), 4)
    hr_test20 = round(np.mean(HR_20), 4)
    ndcg_test20 = round(np.mean(NDCG_20), 4)
    hr_test30 = round(np.mean(HR_30), 4)
    ndcg_test30 = round(np.mean(NDCG_30), 4)
    hr_test40 = round(np.mean(HR_40), 4)
    ndcg_test40 = round(np.mean(NDCG_40), 4)
    hr_test50 = round(np.mean(HR_50), 4)
    ndcg_test50 = round(np.mean(NDCG_50), 4)
    str_print_evl2 = "attacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
    str_print_evl4 = "attacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
    str_print_evl6 = "attacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
    str_print_evl8 = "attacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
    str_print_evl10 = "attacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
    str_print_evl20 = "attacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
    str_print_evl30 = "attacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
    str_print_evl40 = "attacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
    str_print_evl50 = "attacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
    print("-----------------------------------------------------------------")
    print(str_print_evl2)
    print(str_print_evl4)
    print(str_print_evl6)
    print(str_print_evl8)
    print(str_print_evl10)
    print(str_print_evl20)
    print(str_print_evl30)
    print(str_print_evl40)
    print(str_print_evl50)

    HR_2, NDCG_2 = [], []
    HR_4, NDCG_4 = [], []
    HR_6, NDCG_6 = [], []
    HR_8, NDCG_8 = [], []
    HR_10, NDCG_10 = [], []
    HR_20, NDCG_20 = [], []
    HR_30, NDCG_30 = [], []
    HR_40, NDCG_40 = [], []
    HR_50, NDCG_50 = [], []
    set_all = set(range(item_size))
    # all test set length for attacked users
    length = 0
    unattacked_users = set(range(user_size)) - set(attacked_users)
    for u_i in unattacked_users:
        fake_test_user_list = set(test_user_list[u_i]) - set(new_fake_train_user_list[u_i])
        item_i_list = list(fake_test_user_list)
        index_end_i = len(item_i_list)
        length += index_end_i
        if index_end_i != 0:
            item_j_list = list(set_all - set(new_fake_train_user_list[u_i]) - set(fake_test_user_list))
            item_i_list.extend(item_j_list)
            pre_one = all_pre[u_i][item_i_list]
            # indices2 = largest_indices(pre_one, 2)
            # indices4 = largest_indices(pre_one, 4)
            # indices6 = largest_indices(pre_one, 6)
            # indices8 = largest_indices(pre_one, 8)
            # indices10 = largest_indices(pre_one, 10)
            # indices20 = largest_indices(pre_one, 20)
            # indices30 = largest_indices(pre_one, 30)
            # indices40 = largest_indices(pre_one, 40)
            indices50 = largest_indices(pre_one, 50)
            indices2 = list(indices50[0][:2])
            indices4 = list(indices50[0][:4])
            indices6 = list(indices50[0][:6])
            indices8 = list(indices50[0][:8])
            indices10 = list(indices50[0][:10])
            indices20 = list(indices50[0][:20])
            indices30 = list(indices50[0][:30])
            indices40 = list(indices50[0][:40])
            indices50 = list(indices50[0])
            hr_t2, ndcg_t2 = hr_ndcg(indices2, index_end_i, 2)
            hr_t4, ndcg_t4 = hr_ndcg(indices4, index_end_i, 4)
            hr_t6, ndcg_t6 = hr_ndcg(indices6, index_end_i, 6)
            hr_t8, ndcg_t8 = hr_ndcg(indices8, index_end_i, 8)
            hr_t10, ndcg_t10 = hr_ndcg(indices10, index_end_i, 10)
            hr_t20, ndcg_t20 = hr_ndcg(indices20, index_end_i, 20)
            hr_t30, ndcg_t30 = hr_ndcg(indices30, index_end_i, 30)
            hr_t40, ndcg_t40 = hr_ndcg(indices40, index_end_i, 40)
            hr_t50, ndcg_t50 = hr_ndcg(indices50, index_end_i, 50)
            HR_2.append(hr_t2)
            NDCG_2.append(ndcg_t2)
            HR_4.append(hr_t4)
            NDCG_4.append(ndcg_t4)
            HR_6.append(hr_t6)
            NDCG_6.append(ndcg_t6)
            HR_8.append(hr_t8)
            NDCG_8.append(ndcg_t8)
            HR_10.append(hr_t10)
            NDCG_10.append(ndcg_t10)
            HR_20.append(hr_t20)
            NDCG_20.append(ndcg_t20)
            HR_30.append(hr_t30)
            NDCG_30.append(ndcg_t30)
            HR_40.append(hr_t40)
            NDCG_40.append(ndcg_t40)
            HR_50.append(hr_t50)
            NDCG_50.append(ndcg_t50)
    hr_test2 = round(np.mean(HR_2), 4)
    ndcg_test2 = round(np.mean(NDCG_2), 4)
    hr_test4 = round(np.mean(HR_4), 4)
    ndcg_test4 = round(np.mean(NDCG_4), 4)
    hr_test6 = round(np.mean(HR_6), 4)
    ndcg_test6 = round(np.mean(NDCG_6), 4)
    hr_test8 = round(np.mean(HR_8), 4)
    ndcg_test8 = round(np.mean(NDCG_8), 4)
    hr_test10 = round(np.mean(HR_10), 4)
    ndcg_test10 = round(np.mean(NDCG_10), 4)
    hr_test20 = round(np.mean(HR_20), 4)
    ndcg_test20 = round(np.mean(NDCG_20), 4)
    hr_test30 = round(np.mean(HR_30), 4)
    ndcg_test30 = round(np.mean(NDCG_30), 4)
    hr_test40 = round(np.mean(HR_40), 4)
    ndcg_test40 = round(np.mean(NDCG_40), 4)
    hr_test50 = round(np.mean(HR_50), 4)
    ndcg_test50 = round(np.mean(NDCG_50), 4)
    str_print_evl2 = "unattacked user top k = 2 \t" + " hit:" + str(hr_test2) + ' ndcg:' + str(ndcg_test2)
    str_print_evl4 = "unattacked user top k = 4 \t" + " hit:" + str(hr_test4) + ' ndcg:' + str(ndcg_test4)
    str_print_evl6 = "unattacked user top k = 6 \t" + " hit:" + str(hr_test6) + ' ndcg:' + str(ndcg_test6)
    str_print_evl8 = "unattacked user top k = 8 \t" + " hit:" + str(hr_test8) + ' ndcg:' + str(ndcg_test8)
    str_print_evl10 = "unattacked user top k = 10 \t" + " hit:" + str(hr_test10) + ' ndcg:' + str(ndcg_test10)
    str_print_evl20 = "unattacked user top k = 20 \t" + " hit:" + str(hr_test20) + ' ndcg:' + str(ndcg_test20)
    str_print_evl30 = "unattacked user top k = 30 \t" + " hit:" + str(hr_test30) + ' ndcg:' + str(ndcg_test30)
    str_print_evl40 = "unattacked user top k = 40 \t" + " hit:" + str(hr_test40) + ' ndcg:' + str(ndcg_test40)
    str_print_evl50 = "unattacked user top k = 50 \t" + " hit:" + str(hr_test50) + ' ndcg:' + str(ndcg_test50)
    print("-----------------------------------------------------------------")
    print(str_print_evl2)
    print(str_print_evl4)
    print(str_print_evl6)
    print(str_print_evl8)
    print(str_print_evl10)
    print(str_print_evl20)
    print(str_print_evl30)
    print(str_print_evl40)
    print(str_print_evl50)
    print("==================================================================")


add_ratio = 0.8


def project_tensor(pre_cpu, attacked_users_, train_user_list, item_size, user_size, sub_items):
    '''

    :param pre_cpu: A matrix which maximizes the recommendation loss
    :param attacked_users_: user ids who are attacked.
    :param train_user_list: which items have already been interacted by the user
    :param item_size: item size
    :param user_size: user size
    :return:
    '''
    fake_train_user_list = copy.deepcopy(train_user_list)
    added_pair = []
    if sub_items is not None:
        set_all = set(sub_items)
    else:
        set_all = set(range(item_size))
    for u_i in range(len(attacked_users_)):
        user = attacked_users_[u_i]
        item_i_list = list(set_all - set(fake_train_user_list[user]))
        pre_one = pre_cpu[u_i][item_i_list]
        add_length = int(add_ratio*(len(fake_train_user_list[user]))) + 1
        if add_length != 0:
            indices = largest_indices(pre_one, add_length)
            indices = list(indices[0])
            for item in indices:
                item_id = item_i_list[item]
                fake_train_user_list[user].add(item_id)
                added_pair.append([user,item_id])
    train_set_len = 0
    for u_i in range(user_size):
        train_set_len += len(fake_train_user_list[u_i])
    added_pair = np.array(added_pair)
    return added_pair, fake_train_user_list, train_set_len


def mult_ce_loss(data, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    EPSILON = 1e-12
    res = instance_loss / (instance_data + EPSILON)
    return res


def mse_loss(data, logits, weight):
    """Mean square error loss."""
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits)**2
    return res.sum(1)


def WMW_loss_sigmoid(logits_topK, logits_target_items, offset=0.01):
    loss = 0
    for i in range(logits_target_items.shape[1]):
        cur_target_logits = logits_target_items[:, i].reshape(-1, 1)
        x = (logits_topK - cur_target_logits) / offset
        g = torch.sigmoid(x)
        loss += torch.sum(g)
    return loss


def WMW_loss(logits_topK, logits_target_items, offset=0.01):
    loss = 0
    for i in range(logits_target_items.shape[1]):
        cur_target_logits = logits_target_items[:, i].reshape(-1, 1)
        x = logits_topK - cur_target_logits
        x = -1.0 * x / offset
        g = 1.0 / (1 + torch.exp(x))
        loss += torch.sum(g)
    return loss


def sparse2tensor(sparse_data):
    return torch.FloatTensor(sparse_data.toarray())


def tensor2sparse(tensor):
    return sparse.csr_matrix(tensor.detach().cpu().numpy())


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def sim_matrix(a, b, eps=1e-6):
    """
    added eps for numerical stability
    a: M*E,
    a.norm(dim=1): M
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.sparse.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

if __name__ == '__main__':
    if choice == 0:
        # train_clustering_noise()
        train_attack()
    elif choice== 1:
        PATH_MODEL = './preprocessed/RND_'+str(sign)+'.npy'
        print(PATH_MODEL)
        evaluate(PATH_MODEL)
    elif choice== 2:
        PATH_MODEL = './preprocessed/RND_'+str(sign)+'.npy'
        print(PATH_MODEL)
        evaluate_gcn(PATH_MODEL)
    else:
        PATH_MODEL = './preprocessed/RND_' + str(sign) + '.npy'
        # PATH_MODEL = './preprocessed/fake_attackC_' + str(sign) + '.npy'
        print(PATH_MODEL)
        evaluate_itemcf(PATH_MODEL)