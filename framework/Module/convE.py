# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import Parameter as Param
from torch.utils.data import DataLoader


class ConvE(torch.nn.Module):
    """
	Proposed method in the paper. Refer Section 6 of the paper for mode details

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model

	Returns
	-------
	The InteractE model instance

	"""

    def __init__(self,
                 ent_tot,
                 rel_tot,
                 embedding_dim,
                 input_drop,
                 feature_map_drop,
                 hidden_drop,
                 reshape_high,
                 reshape_wide,
                 filter_num):
        super(ConvE, self).__init__()
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.high=reshape_high
        self.wide=reshape_wide
        self.ent_embed = torch.nn.Embedding(self.ent_tot, embedding_dim, padding_idx=None)
        # print(ent_tot)
        xavier_normal_(self.ent_embed.weight.data)
        self.rel_embed = torch.nn.Embedding(self.rel_tot*2, embedding_dim, padding_idx=None);
        xavier_normal_(self.rel_embed.weight.data)

        self.inp_drop = torch.nn.Dropout(input_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)

        self.bceloss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0)

        # bn层：在卷积层之后，去除不稳定
        self.bn1 = torch.nn.BatchNorm2d(filter_num*1)

        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        # self.fc = torch.nn.Linear(1245184, self.p.embed_dim)
        # flattened_size = (self.wide * 2 - 3 + 1) * \
        #                  (self.high - 3 + 1) * 32
        self.fc = torch.nn.Linear(9728, embedding_dim)
        self.register_parameter('bias', Parameter(torch.zeros(ent_tot)))
    def forward(self, sub, rel):
        sub_emb = self.ent_embed(sub).view(-1, 1, self.high, self.wide)
        rel_emb = self.rel_embed(rel).view(-1, 1, self.high, self.wide)
        # print(self.ent_embed.weight.shape)
        stack_inp = torch.cat([sub_emb, rel_emb], 2)
        # print("stack_input",stack_inp.shape)
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(1)
        x = self.fc(x)
        # print(2)
        x.to(torch.device('cuda'))
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        pred = torch.sigmoid(x)
        # print(pred.size())
        return pred
