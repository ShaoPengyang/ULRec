import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
lamada = 0.0001


class NCE(nn.Module):
    def __init__(self, factor_num):
        super(NCE, self).__init__()
        self.layer = nn.Sequential(
                  nn.Linear(factor_num, int(factor_num/2)),
                  nn.ReLU(),
                  nn.Linear(int(factor_num/2), factor_num),
                  nn.Sigmoid()
                )
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        output = self.layer(input)
        loss = self.bce(output,target)
        return loss

    def predict(self, input):
        output = self.layer(input)
        return output


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        # nn.init.constant_(self.embed_user.weight, val=0.1)
        # nn.init.constant_(self.embed_item.weight, val=0.1)

    def forward(self, user, item_i, item_j, return_matrix = False):
        if return_matrix==True:
            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight
            return torch.mm(users_embedding, items_embedding.t())
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)
        # positive
        prediction_i = (user * item_i).sum(dim=-1)
        # negative
        prediction_j = (user * item_j).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2 + item_j ** 2).mean()
        bpr_loss = -(prediction_i - prediction_j).sigmoid().log()
        bpr_loss = bpr_loss.mean()
        loss = bpr_loss + l2_regulization
        return loss

    def get_embedding(self):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        return users_embedding, items_embedding


# add weights for each user-item pair
class WBPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(WBPR, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.rmse = nn.MSELoss()

    def forward(self, user, item_i, item_j, intention):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        # item_j = self.embed_item(item_j)
        intention = intention.float()
        prediction_i = (user * item_i).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2).mean()
        loss2 = self.rmse(intention,prediction_i)
        loss = loss2 + l2_regulization
        return loss

    def get_embedding(self):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        return users_embedding, items_embedding


class LRGCCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train):
        super(LRGCCF, self).__init__()
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        self.user_item_matrix = sparse_u_i
        self.item_user_matrix = sparse_i_u
        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]
        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

    def forward(self, user, item_i, item_j):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding
        user = F.embedding(user, gcn_users_embedding)
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2 + item_j ** 2).mean()
        loss2 = -(prediction_i - prediction_j).sigmoid().log()
        loss2 = loss2.mean()
        loss = loss2 + l2_regulization
        return loss
    
    def get_embedding(self):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding
        return gcn_users_embedding, gcn_items_embedding

    def get_origin(self):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        return users_embedding, items_embedding

class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dims):
        super(WeightedMF, self).__init__()

        hidden_dims = hidden_dims
        if len(hidden_dims) > 1:
            raise ValueError("WMF can only have one latent dimension.")

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]

        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        # users
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            # pdb.set_trace()
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())

    def get_embedding(self):
        return self.P.clone().detach(), self.Q.clone().detach()


class WeightedMF_Graph(nn.Module):
    def __init__(self, n_users, n_items, hidden_dims):
        super(WeightedMF_Graph, self).__init__()
        hidden_dims = hidden_dims
        if len(hidden_dims) > 1:
            raise ValueError("WMF can only have one latent dimension.")
        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]
        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.01))
        # users
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.01))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        users_embedding = self.P
        items_embedding = self.Q
        gcn1_users_embedding = (torch.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding), -1)
        gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding), -1)
        # gcn_users_embedding = gcn1_users_embedding #+ gcn2_users_embedding
        # gcn_items_embedding = gcn1_items_embedding #+ gcn2_items_embedding
        if user_id is None and item_id is None:
            return torch.mm(gcn_users_embedding, gcn_items_embedding.t())
        if user_id is not None:
            # pdb.set_trace()
            return torch.mm(gcn_users_embedding[[user_id]], gcn_items_embedding.t())
        if item_id is not None:
            return torch.mm(gcn_users_embedding, gcn_items_embedding[[item_id]].t())

    def get_embedding(self):
        users_embedding = self.P
        items_embedding = self.Q
        gcn1_users_embedding = (torch.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding), -1)
        gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding), -1)
        # gcn_users_embedding = users_embedding + gcn1_users_embedding + gcn2_users_embedding
        # gcn_items_embedding = items_embedding + gcn1_items_embedding + gcn2_items_embedding
        # gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding), -1)
        # gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding), -1)
        return torch.mm(gcn_users_embedding, gcn_items_embedding.t())

    def set_Adjacency(self, Adjacency):
        self.Adjacency = Adjacency
        self.d_i_train = 1.0 / (self.Adjacency.sum(-1) + 1.0)
        self.d_j_train = 1.0 / (self.Adjacency.sum(0) + 1.0)
        self.user_item_matrix = self.Adjacency * torch.sqrt(self.d_i_train.reshape(-1, 1)) * torch.sqrt(self.d_j_train.reshape(1, -1))
        self.item_user_matrix = self.user_item_matrix.t()#.to_sparse()
        self.d_i_train = self.d_i_train.reshape(-1,1).repeat(1,self.dim)
        self.d_j_train = self.d_j_train.reshape(-1,1).repeat(1,self.dim)


class GCN(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(GCN, self).__init__()
        self.dim = factor_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        gcn1_users_embedding = (torch.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding),-1)
        gcn_items_embedding = torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding),-1)
        user = F.embedding(user, gcn_users_embedding)
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        l2_regulization = lamada * (user ** 2).mean() + lamada * (item_i ** 2 + item_j ** 2).mean()
        loss2 = -(prediction_i - prediction_j).sigmoid().log()
        loss2 = loss2.mean()
        loss = loss2 + l2_regulization
        return loss

    def get_embedding(self):
        users_embedding = self.embed_user.weight
        items_embedding = self.embed_item.weight
        gcn1_users_embedding = (torch.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
            self.d_i_train))
        gcn1_items_embedding = (torch.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
            self.d_j_train))
        gcn2_users_embedding = (torch.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
            self.d_i_train))
        gcn2_items_embedding = (torch.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
            self.d_j_train))
        gcn_users_embedding = torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding), -1)
        gcn_items_embedding = torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding), -1)
        return gcn_users_embedding, gcn_items_embedding

    def set_Adjacency(self, Adjacency):
        self.Adjacency = Adjacency
        self.user_item_matrix = self.Adjacency#.to_sparse()
        self.item_user_matrix = self.Adjacency.t()#.to_sparse()
        self.d_i_train = 1.0 / (self.Adjacency.sum(-1) + 1.0)
        self.d_j_train = 1.0 / (self.Adjacency.sum(0) + 1.0)
        pdb.set_trace()
        self.d_i_train = self.d_i_train.reshape(-1,1).repeat(1,self.dim)
        self.d_j_train = self.d_j_train.reshape(-1,1).repeat(1,self.dim)
        # pdb.set_trace()