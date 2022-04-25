import torch
import torch.nn as nn
import torch.nn.functional as F


class BUIR_ID(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, momentum):
        super(BUIR_ID, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.momentum = momentum

        self.uo_encoder = nn.Embedding(self.num_users, hidden_dim)
        self.ut_encoder = nn.Embedding(self.num_users, hidden_dim)
        self.io_encoder = nn.Embedding(self.num_items, hidden_dim)
        self.it_encoder = nn.Embedding(self.num_items, hidden_dim)
        for l in [self.uo_encoder, self.ut_encoder, self.io_encoder, self.it_encoder]:
            nn.init.xavier_normal_(l.weight.data)

        self.predictor = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.predictor.weight.data)
        nn.init.xavier_normal_(self.predictor.bias.data)

        # init target
        for param_o, param_t in zip(self.uo_encoder.parameters(), self.ut_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.io_encoder.parameters(), self.it_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(self.uo_encoder.parameters(), self.ut_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

        for param_o, param_t in zip(self.io_encoder.parameters(), self.it_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def forward(self, inputs):
        user, item = inputs

        u_online = self.predictor(self.uo_encoder(user))
        u_target = self.ut_encoder(user)
        i_online = self.predictor(self.io_encoder(item))
        i_target = self.it_encoder(item)

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online = self.uo_encoder.weight
        i_online = self.io_encoder.weight
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def get_loss(self, output):
        output = list(output)
        for i in range(len(output)):
            output[i] = F.normalize(output[i], dim=-1)
        o_user, t_user, o_item, t_item = output
        loss = 2 * ((1 - (o_user * t_item).sum(dim=-1)) + (1 - (o_item * t_user).sum(dim=-1)))
        return loss.mean()
