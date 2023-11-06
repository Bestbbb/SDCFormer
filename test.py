import torch
import torch.nn as nn
import pytorch_lightning as pl


print(pl.LightningModule)

x = torch.randn(32, 96, 512)


class myLinear(pl.LightningModule):
    def __init__(self):
        super(myLinear, self).__init__()
        self.linear = nn.Linear(512, 512)

    def forward(self,x):
        y = self.linear(x)
        return y


model = myLinear()
x_out = model(x).to('cuda')

print(x_out.shape)

queries = torch.ones(32,96,8,64)
a = queries.permute(0, 2, 3, 1).contiguous()
q_fft = torch.fft.rfft(a, dim=-1)
q2_fft = torch.fft.fft(a, dim=-1)
print("rfft",q_fft)
print("fft",q2_fft)
q2 = torch.fft.ifft(q2_fft,dim=-1)
q = torch.fft.irfft(q_fft,dim=-1)
print(q.shape)
print(q2.shape)
mean_value= torch.ones(32,96)
index = torch.topk(torch.mean(mean_value, dim=0), 13, dim=-1)[1]
weights = torch.stack([mean_value[:, index[i]] for i in range(13)], dim=-1)
result = torch.softmax(weights,dim=-1)
sum = torch.sum(result,dim=-1,keepdim=True)
print(sum)


t = torch.arange(6).reshape(1,2,3)
a = t[:,:2]
print(a)

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
print(output.shape)
input = torch.randn(2,5,96)
linear = nn.Linear(96,192)
output = linear(input)
print(output.shape)


x = torch.randn(32,96,21,6)
moving_mean = torch.randn(32,96,21,6)
moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(x), dim=1)
print(moving_mean.shape)
