
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class SiNE(nn.Module):
    def __init__(self, num_nodes, dim1, dim2):
        super(SiNE, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(num_nodes + 1, dim1)
        # first linear
        self.layer11 = nn.Linear(dim1, dim2, bias=False)
        self.layer12 = nn.Linear(sim1, dim2, bias=False)
        self.bias1 = Parameter(torch.zeros(1))
        # second linear(end)
        self.layer2 = nn.Linear(dim2, 1, bias=False)
        self.bias2 = Parameter(torch.zeros(1))
        
        self.tanh = nn.Tanh()
        self.register_parameter('bias1', self.bias1)
        self.register_parameter('bias2', self.bias2)
    
    def forward(self, xi, xj, xk, delta):
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        z11 = self.tanh(self.layer11(i_emb) + self.layer12(j_emb) + self.bias1)
        z12 = self.tanh(self.layer11(i_emb) + self.layer12(k_emb) + self.bias1)
        
        f_pos = self.tanh(self.layer2(z11) + self.bias2)
        f_neg = self.tanh(self.layer2(z12) + self.bias2)
        
        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, f_pos + delta - f_neg)
        loss = torch.sum(loss)

        return loss

