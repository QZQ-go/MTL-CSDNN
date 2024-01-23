"""
计算多头深度学习的每个决策头的相似度
"""
import pandas
import torch
from dnn_net import MultiTaskDnn


model = MultiTaskDnn()
model.load_state_dict(torch.load('model_para.pt'))

res_dict = {k1: {k2: 0 for k2 in model.multi_head_dict.keys()} for k1 in model.multi_head_dict.keys()}

for k1 in model.multi_head_dict.keys():
    for k2 in model.multi_head_dict.keys():
        # torch.Size([1, 64])
        param1 = model.multi_head_dict[k1][0].weight
        param2 = model.multi_head_dict[k2][0].weight
        pdist = torch.nn.PairwiseDistance(p=2)  # p=2就是计算欧氏距离，p=1就是曼哈顿距离
        out_put = pdist(param1, param2).item()
        res_dict[k1][k2] = out_put

df = pandas.DataFrame(res_dict)
df.to_csv('head_dist.csv')
