import torch
import torch.nn as nn
import torch.nn.functional as F

class KLAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(KLAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, num_segments, input_dim)
        query = self.query(x)  # shape: (batch_size, num_segments, hidden_dim)
        key = self.key(x)  # shape: (batch_size, num_segments, hidden_dim)
        value = self.value(x)  # shape: (batch_size, num_segments, hidden_dim)
        
        num_segments = x.shape[1]
        kl_divs = torch.zeros(32,num_segments, num_segments)
        for i in range(num_segments):
            for j in range(num_segments):
                    if i==j==0:
                        print(query[:,i].shape)
                        kl_divs[:,i, j] = self.kl_div(query[:, i], key[:, j])
        
        attn_weights = self.softmax(-kl_divs)  # shape: (num_segments, num_segments)
        attn_output = torch.matmul(attn_weights, value)  # shape: (batch_size, num_segments, hidden_dim)
        return attn_output
    
    def kl_div(self, query, key):
        p = F.softmax(query, dim=-1)
        q = F.softmax(key, dim=-1)
        print((p * (torch.log(p) - torch.log(q))).sum(dim=-1).shape)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)




class KLAttention2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(KLAttention2, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        query = x
        key = x
        value = x 
        # x shape: (batch_size, num_segments, input_dim)
        # query = self.query(x)  # shape: (batch_size, num_segments, hidden_dim)
        # key = self.key(x)  # shape: (batch_size, num_segments, hidden_dim)
        # value = self.value(x)  # shape: (batch_size, num_segments, hidden_dim)
        
        num_segments = x.shape[1]
        kl_divs = torch.zeros(32,num_segments, num_segments)
        for b in range(32):
            for i in range(num_segments):
                for j in range(num_segments):
                    if i != j:
                        kl_divs[b,i, j] = self.kl_div(query[b, i], key[b, j])

        print(kl_divs)
        # query = self.query(x)  # shape: (batch_size, num_segments, hidden_dim)
        # key = self.key(x)  # shape: (batch_size, num_segments, hidden_dim)
        # value = self.value(x)  # shape: (batch_size, num_segments, hidden_dim)
        
        num_segments = x.shape[1]
        kl_divs = torch.zeros(32,num_segments, num_segments)
        for i in range(num_segments):
            for j in range(num_segments):
                if i != j:
                    kl_divs[:,i, j] = self.kl_div(query[:, i], key[:, j])
        
        print(kl_divs)

        
        attn_weights = self.softmax(-kl_divs)  # shape: (num_segments, num_segments)
        attn_output = torch.matmul(attn_weights, value)  # shape: (batch_size, num_segments, hidden_dim)
        return attn_output
    
    def kl_div(self, query, key):
        p = F.softmax(query, dim=-1)
        q = F.softmax(key, dim=-1)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)








model = KLAttention(512,512)
q = torch.randn(32,11,512)
k = torch.randn(32,11,512)
v = torch.randn(32,11,512)
res = model(q)

