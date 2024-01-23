import copy
import pandas
import torch.nn.functional as F
from data_loader import DISEASE_CSV_PATH
from collections import OrderedDict
from torch import torch, nn

disease_df = pandas.read_csv(DISEASE_CSV_PATH, index_col=0)
disease_name = disease_df.columns.tolist()
for col in ["视力障碍", "idnum"]:
    if col in disease_name:
        disease_name.remove(col)


class MultiTaskDnn(nn.Module):
    def __init__(self,
                 input_size=151,
                 use_shap=False):
        super().__init__()
        self.use_shap = use_shap
        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_size, 512)),
                    ("relu1", nn.ReLU()),

                    ("linear2", nn.Linear(512, 256)),
                    ("relu2", nn.ReLU()),

                    ("linear3", nn.Linear(256, 64)),
                    ("relu3", nn.ReLU()),
                ]
            )
        )

        self.head_pattern = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(64, 1)),
                    ("Sigmoid", nn.Sigmoid()),
                ]
            )
        )

        self.multi_head_dict = nn.ModuleDict({
            dse_name: copy.deepcopy(self.head_pattern) for dse_name in disease_name
        })

    def forward(self, x):
        """
        :param x: 由于需要进行分头训练，所以数据是按照字典的方式储存的，需要转译一下
        :return:
        """
        head_res = {}
        for dse_name in disease_name:
            _x = self.body(x[dse_name])
            _x = self.multi_head_dict[dse_name](_x)
            head_res.update({dse_name: _x})
        if not self.use_shap:
            return head_res
        else:
            return sum(i[1] for i in head_res.items())


class GlobalTaskDnn(nn.Module):
    def __init__(self, input_size=151):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(input_size, 512)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(512, 256)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(256, 64)),
                    ("relu3", nn.ReLU()),

                    # 决策层
                    ("linearX", nn.Linear(64, 1)),
                    ("Sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        return self.net(x)


class Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linearX", nn.Linear(151, 1)),
                    ("Sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskDnnTest(MultiTaskDnn):
    """
    父类作为多任务学习模型，每次都会传15种疾病进行训练，但是这对可视化来说比较难操作
    子类将只向前传播单种疾病的数据，从而让可视化成为可能
    """

    def __init__(self, target_disease):
        super().__init__()
        self.target_disease = target_disease

    def forward(self, x):
        _x = self.body(x)
        _x = self.multi_head_dict[self.target_disease](_x)
        return _x


class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(151, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, _input_size, _num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(_input_size, _num_classes)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class CNN1D(nn.Module):
    """
    参考这个模型，来源：
    https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/inference/inference.ipynb
    写了一份自己版本的cnn，主要借鉴了一下可以通过第一层的密集链接层来让模型学习正确的特征排序问题
    从而解决了1DCNN只能解决时序问题的弊端
    """

    def __init__(self, num_features=151, num_targets=1, hidden_size=512):
        super(MyCNN1D, self).__init__()

        self.cha_1 = 128
        self.cha_2 = 256
        self.cha_3 = 256
        self.cha_1_reshape = int(hidden_size / self.cha_1)
        self.cha_po_1 = int(hidden_size / self.cha_1 / 2)
        self.cha_po_2 = int(hidden_size / self.cha_1 / 2 / 2) * self.cha_3

        # 第一层网络，经过一层dense层，将151的特征扩张至512
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        # 第二层网络开始进行卷积，将输入的512个个特征的一维特征
        self.batch_norm_c1 = nn.BatchNorm1d(self.cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_1, self.cha_2, kernel_size=5, stride=1, padding=2, bias=False),
            dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=self.cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_3, kernel_size=5, stride=1, padding=2, bias=True),
            dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(self.cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(self.cha_po_2, num_targets))

    def forward(self, x):
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        x = torch.sigmoid(x)
        return x


class MyCNN1D(nn.Module):
    """
    https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/inference/inference.ipynb
    参考这个模型，进行了一些修改
    """
    def __init__(self, num_features=151, num_targets=1, hidden_size=512):
        super(MyCNN1D, self).__init__()

        self.cha_1 = 128
        self.cha_2 = 256
        self.cha_3 = 256
        self.cha_1_reshape = int(hidden_size / self.cha_1)
        self.cha_po_1 = int(hidden_size / self.cha_1 / 2)
        self.cha_po_2 = int(hidden_size / self.cha_1 / 2 / 2) * self.cha_3

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_1, self.cha_2, kernel_size=5, stride=1, padding=2, bias=False),
            dim=None)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=self.cha_po_1)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None)
        self.conv2_1 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None)

        self.conv2_2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_3, kernel_size=5, stride=1, padding=2, bias=True),
            dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.flt = nn.Flatten()
        self.dense3 = nn.utils.weight_norm(nn.Linear(self.cha_po_2, num_targets))

    def forward(self, x):
        x = F.celu(self.dense1(x), alpha=0.06)
        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)
        x = F.relu(self.conv1(x))
        x = self.ave_po_c1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.max_po_c2(x)
        x = self.flt(x)
        x = self.dense3(x)
        x = torch.sigmoid(x)
        return x
