import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BUIR_NB(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, adj_mat, momentum, num_layers=3, drop_out=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.adj_mat = adj_mat
        self.momentum = momentum
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.o_encoder = LGCN_Encoder(num_users, num_items, adj_mat, hidden_dim, num_layers, drop_out)
        self.t_encoder = LGCN_Encoder(num_users, num_items, adj_mat, hidden_dim, num_layers, drop_out)

        self.F_layer = nn.Linear(hidden_dim, hidden_dim)

        # init t_encoder
        for o_param, t_param in zip(self.o_encoder.parameters(), self.t_encoder.parameters()):
            t_param.data.copy_(o_param.data)
            t_param.requires_grad = False

    def _update_target(self):
        for o_param, t_param in zip(self.o_encoder.parameters(), self.t_encoder.parameters()):
            t_param.data = t_param.data * self.momentum + o_param.data * (1. - self.momentum)

    def forward(self, inputs):
        o_user, o_item = self.o_encoder(inputs)
        t_user, t_item = self.o_encoder(inputs)

        o_user = self.F_layer(o_user)
        t_user = self.F_layer(t_user)
        return o_user, t_user, o_item, t_item

    @torch.no_grad()
    def get_embedding(self):
        user, item = self.o_encoder.get_embeddings()
        return self.F_layer(user), user, self.F_layer(item), item

    def get_loss(self, output):

        output = list(output)
        for i in range(len(output)):
            output[i] = F.normalize(output[i], dim=-1)

        o_user, t_user, o_item, t_item = output

        loss = 2 * ((1 - (o_user * t_item).sum(dim=-1)) + (1 - (o_item * t_user).sum(dim=-1)))
        return loss.mean()

class LGCN_Encoder(nn.Module):
    def __init__(self, num_users, num_items, adj_mat, hidden_dim, num_layers=3, drop_out=0):

        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.adj_mat = torch.Tensor(adj_mat)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # TODO : sparse dropout?
        self.drop_out = nn.Dropout(p=drop_out)

        self.u_embed = nn.Parameter(torch.zeros(self.num_users, self.hidden_dim))
        self.i_embed = nn.Parameter(torch.zeros(self.num_items, self.hidden_dim))

        nn.init.xavier_uniform_(self.u_embed)
        nn.init.xavier_uniform_(self.i_embed)

    def forward(self, inputs):

        embeddings = torch.cat([self.u_embed, self.i_embed], 0)
        mat = self.drop_out(self.adj_mat)

        all_embeddings = [embeddings]

        for layer in range(self.num_layers):
            embeddings = torch.matmul(mat, embeddings)
            all_embeddings.append(embeddings)

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        users, items = inputs
        user_embeddings = all_embeddings[:self.num_users, :][users]
        item_embeddings = all_embeddings[self.num_users:, :][items]

        return user_embeddings, item_embeddings
    
    @torch.no_grad()
    def get_embeddings(self):

        embeddings = torch.cat([self.u_embed, self.i_embed], 0)
        mat = self.drop_out(self.adj_mat)

        all_embeddings = [mat]

        for layer in range(self.num_layers):
            mat = torch.matmul(mat, ego_embeddings)
            all_embeddings.append(mat)

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        return all_embeddings[:self.num_users, :], all_embeddings[self.num_users:, :]



