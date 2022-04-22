import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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



