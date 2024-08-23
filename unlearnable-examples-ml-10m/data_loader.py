import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import pdb
from copy import deepcopy
import pickle


def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair


def readD(set_matrix, num_):
    user_d = []
    # i =0,1,2...6013
    for i in range(num_):
        len_set = 1.0 / (len(set_matrix[i]) + 1)
        user_d.append(len_set)
    return user_d


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def readTrainSparseMatrix(set_matrix, is_user, u_d, i_d, user_num, item_num):
    user_items_matrix_i = []
    user_items_matrix_v = []
    if is_user:
        d_i = u_d
        d_j = i_d
        num_ = user_num
        user_items_matrix_i.append([user_num-1, item_num-1])
        user_items_matrix_v.append(0)
    else:
        d_i = i_d
        d_j = u_d
        num_ = item_num
        user_items_matrix_i.append([item_num-1, user_num-1])
        user_items_matrix_v.append(0)
    for i in range(num_):
        for j in set_matrix[i]:
            user_items_matrix_i.append([i, j])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i = torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v = torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)


class TripletUniformPair(Dataset):
    def __init__(self, num_item, user_list, num_ng, set_len):
        super(TripletUniformPair, self).__init__()
        self.num_item = num_item
        self.user_list = user_list
        self.num_ng = num_ng
        self.set_len = set_len

    def ng_sample(self):
        self.uij = []
        for user in range(len(self.user_list)):
            positive_list = list(self.user_list[user])
            # if 3532 in positive_list:
            #     pdb.set_trace()
            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j = np.random.randint(self.num_item)
                    while item_j in positive_list:
                        item_j = np.random.randint(self.num_item)
                    self.uij.append([user, item_i, item_j])

    def __getitem__(self, idx):
        uij = self.uij
        u = uij[idx][0]
        i = uij[idx][1]
        j = uij[idx][2]
        return u, i, j

    def __len__(self):
        return self.set_len * self.num_ng

class TripletUniformPair2(Dataset):
    def __init__(self, num_item, user_list, target_items, num_ng, set_len):
        super(TripletUniformPair2, self).__init__()
        self.num_item = num_item
        self.user_list = user_list
        self.num_ng = num_ng
        self.target_items = target_items
        self.set_len = set_len

    def ng_sample(self):
        self.uij = []
        for user in range(len(self.user_list)):
            positive_list = list(self.user_list[user])
            # if 3532 in positive_list:
            #     pdb.set_trace()
            for item_i in positive_list:
                target_item = np.random.choice(self.target_items)
                for t in range(self.num_ng):
                    item_j = np.random.randint(self.num_item)
                    while item_j in (set(positive_list) | set(self.target_items)):
                        item_j = np.random.randint(self.num_item)
                    self.uij.append([user, item_i, item_j])
                self.uij.append([user, target_item, item_i])

    def __getitem__(self, idx):
        uij = self.uij
        u = uij[idx][0]
        i = uij[idx][1]
        j = uij[idx][2]
        return u, i, j

    def __len__(self):
        return self.set_len * (self.num_ng + 1)


class TripletUniformPairDict(Dataset):
    def __init__(self, num_item, top_list, train_user_list, num_ng=1, set_len=1000):
        super(TripletUniformPairDict, self).__init__()
        self.num_item = num_item
        self.top_list = top_list
        self.num_ng = num_ng
        self.set_len = set_len
        self.train_user_list = train_user_list

    def ng_sample(self):
        self.uij = []
        for user, item_set in self.top_list.items():
            positive_list = list(set(self.top_list[user].keys())|self.train_user_list[user])
            for item_i, intention in item_set.items():
                for t in range(self.num_ng):
                    item_j = np.random.randint(self.num_item)
                    while item_j in positive_list:
                        item_j = np.random.randint(self.num_item)
                    self.uij.append([user, item_i, item_j])

    def __getitem__(self, idx):
        uij = self.uij
        u = uij[idx][0]
        i = uij[idx][1]
        j = uij[idx][2]
        intention = uij[idx][3]
        return u, i, j, intention

    def __len__(self):
        return self.set_len * self.num_ng

# class TripletUniformPairDict(Dataset):
#     def __init__(self, num_item, top_list, train_user_list, num_ng=1, set_len=1000):
#         super(TripletUniformPairDict, self).__init__()
#         self.num_item = num_item
#         self.top_list = top_list
#         self.num_ng = num_ng
#         self.set_len = set_len
#         self.train_user_list = train_user_list
#
#     def ng_sample(self):
#         self.uij = []
#         for user, item_set in self.user_list.items():
#             positive_list = list(self.user_list[user].keys())
#             for item_i, intention in item_set.items():
#                 for t in range(self.num_ng):
#                     item_j = np.random.randint(self.num_item)
#                     while item_j in positive_list:
#                         item_j = np.random.randint(self.num_item)
#                     self.uij.append([user, item_i, item_j, intention])
#
#     def __getitem__(self, idx):
#         uij = self.uij
#         u = uij[idx][0]
#         i = uij[idx][1]
#         j = uij[idx][2]
#         intention = uij[idx][3]
#         return u, i, j, intention
#
#     def __len__(self):
#         return self.set_len * self.num_ng

# random data

def init_fake_data(user_list, item_size, num_fake_item):
    fake_user_list = deepcopy(user_list)
    all_items = set(range(item_size))
    add_pair_num = 0
    record_txt = open("./record_fake_items_random.txt", 'a')
    for user, item_set in enumerate(user_list):
        negative_items = all_items - item_set
        fake_item_set = random.sample(list(negative_items), num_fake_item)
        new_item_set = item_set | set(fake_item_set)
        fake_user_list[user] = new_item_set
        add_pair_num += num_fake_item
        record_txt.write('user: ' + str(user) + " items:" + str(fake_item_set)+'\n')
    record_txt.flush()
    return fake_user_list, add_pair_num

def get_topK_items(num_fake_item):
    from CF_model import WBPR
    '''
        learning rate: 0.001
        batchsize = 8192*2
        :return:
        '''
    print('----------------loading data--------------------------')
    # Load preprocess data
    with open('./preprocessed/ml-1m_IPW.pickle', 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['user_size'], dataset['item_size']
        train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
    print('-------------getting top B---------------------')
    noise_train_user_list = deepcopy(train_user_list)
    pre_trained_model = WBPR(user_size, item_size, 64).cuda()
    pre_trained_model.train()
    pre_trained_model.load_state_dict(torch.load('./model/ml-1m/t0/wbpr_10.pt'))
    pre_trained_model.eval()
    with torch.no_grad():
        user_e, item_e = pre_trained_model.get_embedding()
    user_e = user_e.cpu().detach().numpy()
    item_e = item_e.cpu().detach().numpy()
    all_pre = np.matmul(user_e, item_e.T)
    # np.save('./preprocessed/all_pre.npy', all_pre)
    all_pre = 0.0 - all_pre
    add_pair_num = 0
    record_txt = open("./record_fake_items_top.txt",'a')
    item_idx_selected = np.load('./preprocessed/idx_selected.npy', allow_pickle=True)
    set_all = set(item_idx_selected)
    for user, item_set in enumerate(train_user_list):
        item_i_list = list(set_all - set(train_user_list[user]))
        pre_one = all_pre[user][item_i_list]
        indices = largest_indices(pre_one, num_fake_item)
        ind = np.array(indices[0])
        item_i_list = np.array(item_i_list)
        fake_item_set = list(item_i_list[ind])
        new_item_set = item_set | set(fake_item_set)
        record_txt.write('user: '+str(user)+" items:"+str(fake_item_set)+'\n')
        noise_train_user_list[user] = new_item_set
        add_pair_num += num_fake_item
    record_txt.flush()
    pdb.set_trace()
    return noise_train_user_list

# method 1
def Top_K_data(all_pre, K=20):
    fake_user_list = deepcopy(user_list)
    all_items = set(range(item_size))
    add_pair_num = 0
    for user, item_set in enumerate(fake_user_list):
        negative_items = all_items - item_set
        fake_item_set = random.sample(list(negative_items), num_fake_item)
        new_item_set = item_set | set(fake_item_set)
        fake_user_list[user] = new_item_set
        add_pair_num += num_fake_item
    return fake_user_list, user_list, add_pair_num