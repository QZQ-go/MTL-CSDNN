import shap
import torch
from torch.utils.data import random_split, DataLoader
from data_loader import SingleDiseaseDataSet
from matplotlib import pyplot as plt
from IPython.display import (display, display_html, display_png, display_svg)
from get_feature_names import get_feature_name_list
from net import MultiTaskDnnTest, disease_name

ft_list = get_feature_name_list()


def my_shap(target_diseases='血脂异常', test_dataset_len=None):
    # 准备数据集
    data_set = SingleDiseaseDataSet(target_diseases=target_diseases)
    train_len = int(round(len(data_set) * 0.8, 0))
    test_len = int(round(len(data_set) * 0.1, 0))
    train_len += len(data_set) - train_len - test_len * 2
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=data_set, lengths=[train_len, test_len, test_len], generator=torch.Generator().manual_seed(5)
    )

    # 准备模型
    model = MultiTaskDnnTest(target_diseases)

    model.load_state_dict(torch.load('mt_dnn_model_para.pt'))
    model.eval()

    train_data_loader = DataLoader(train_dataset,
                                   shuffle=True,
                                   batch_size=100)

    _, data = next(enumerate(train_data_loader))
    inputs: dict = data[0]

    explainer = shap.DeepExplainer(model, inputs)
    if test_dataset_len is not None:
        aimed_len = test_dataset_len
    else:
        aimed_len = len(test_dataset)

    shap_values = explainer.shap_values(torch.Tensor([test_dataset[i][0] for i in range(aimed_len)]))
    shap.initjs()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体
    plt.rcParams['axes.unicode_minus'] = False  # 确保坐标轴负号显示正常
    plt.clf()
    plt.figure()
    shap.summary_plot(shap_values, torch.Tensor([test_dataset[i][0] for i in range(aimed_len)]), show=False, feature_names=ft_list)

    # shap.force_plot(explainer.expected_value, shap_values, matplotlib=True)

    plt.savefig(f'shap/{target_diseases}.png')


if __name__ == '__main__':
    for name in disease_name:
        my_shap(name)
    