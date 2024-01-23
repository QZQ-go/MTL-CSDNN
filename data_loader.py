import copy
import os.path
import random
import numpy
import pandas
import torch
from torch.utils.data import Dataset
import multiprocessing as mul
import pickle


def auto_mul_process(func, data_list, proc_num=None):
    if proc_num is None:
        if len(data_list) < 30:
            proc_num = len(data_list)
        else:
            proc_num = 30
    pool = mul.Pool(proc_num)
    rel = pool.map(func, data_list)
    pool.close()
    pool.join()
    return rel


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(5)

DATA_CSV_PATH = r"csv_file/knn_4_mask2022.csv"
DEATH_CSV_PATH = 'csv_file/death_label.csv'
DISEASE_CSV_PATH = r"csv_file/res 9.21.csv"
CATE_CSV_PATH = r"csv_file/data_cate2.csv"

if os.path.exists(DATA_CSV_PATH):
    tj_data_df = pandas.read_csv(DATA_CSV_PATH, index_col=0)
death_df = pandas.read_csv(DEATH_CSV_PATH, index_col=0)
disease_df = pandas.read_csv(DISEASE_CSV_PATH, index_col=0)
disease_df.drop(columns=["视力障碍"], inplace=True)

cate_df = pandas.read_csv(CATE_CSV_PATH)
numerical_fields = cate_df[cate_df["type"] == "numerical"]["name"].tolist()
categorical_fields = cate_df[cate_df["type"] == "categorical"]["name"].tolist()


def rand_fill_list(_list: list, _len):
    res_list = []
    while len(res_list) < _len:
        if _len - len(res_list) > len(_list):
            res_list.extend(random.sample(_list, len(_list)))
        else:
            res_list.extend(random.sample(_list, _len - len(res_list)))
    return res_list


def _get_data_arr_dict(i):
    def _str_to_arr(t: str):
        t = t.strip("[]")
        t = t.strip()
        _arr_list = t.split(" ")
        while "" in _arr_list:
            _arr_list.remove("")
        return numpy.array([int(i.replace(".", "")) for i in _arr_list])

    sr: pandas.Series = tj_data_df.iloc[i, :]
    id_num = sr.name
    numerical_arr = numpy.array(sr[numerical_fields])
    cate_sr: pandas.Series = sr[categorical_fields].apply(_str_to_arr).tolist()
    categorical_arr = numpy.concatenate(cate_sr)
    final_arr = numpy.concatenate([numerical_arr, categorical_arr])

    return {int(id_num): final_arr}


def save_data_arr_dict():
    """
    在DataFrame中的数据格式是以Series的格式存在，这种格式模型没办法直接利用
    这里将其转化成numpy中的向量格式，由于数据集本身并不大，所以全部读进内存也没什么问题
    """
    _data_dict = {}
    dict_list = auto_mul_process(_get_data_arr_dict, list(range(len(tj_data_df))), 7)
    # dict_list = [_get_data_arr_dict(i) for i in range(len(tj_data_df))]
    for _dict in dict_list:
        _data_dict.update(_dict)
    with open('arr_dict', 'wb') as f:
        pickle.dump(_data_dict, f)


def get_data_arr_dict():
    with open('csv_file/arr_dict', 'rb') as f:
        res = pickle.load(f)
    return res


class DiseaseDataSet(Dataset):
    """
    每个病症对应一个
    返回每个病症的数据样本
    """
    def __init__(self, target_diseases=None):
        self.target_diseases = target_diseases
        # 如果有标记该参数，则专门提供一个疾病的学习样本
        self.data_arr_dict = get_data_arr_dict()
        print("finish init data_dict")
        self.dse_df = disease_df

        # 获取疾病名称的列表
        self.dse_name_list = disease_df.columns.to_list()
        if "idnum" in self.dse_name_list:
            self.dse_name_list.remove("idnum")
        if self.target_diseases is not None:
            self.dse_name_list = [self.target_diseases]

        # 获取死亡的id
        self.death_id = death_df["idnum"].tolist()

        # 获取疾病名称到对应的患者群体id的映射
        self.dse_name_to_id_dict = {
            dse_name: self.dse_df[self.dse_df[dse_name]]["idnum"].tolist()
            for dse_name in self.dse_name_list
        }
        self.max_len = max(
            [len(self.dse_name_to_id_dict[k]) for k in self.dse_name_to_id_dict.keys()]
        )

        # 将所有的id列表补全至同一长度
        self.dse_to_ids_dict = copy.deepcopy(self.dse_name_to_id_dict)
        for k in self.dse_to_ids_dict.keys():
            self.dse_to_ids_dict[k] = rand_fill_list(
                self.dse_name_to_id_dict[k], self.max_len
            )

    def __getitem__(self, index):
        x = {
            dse_name: self.data_arr_dict[self.dse_to_ids_dict[dse_name][index]]
            for dse_name in self.dse_name_list
        }
        y = {dse_name: 1
        if self.dse_to_ids_dict[dse_name][index] in self.death_id else 0
             for dse_name in self.dse_name_list}

        for k in x.keys():
            x[k] = torch.Tensor(x[k].astype('float64'))
        return x, y

    def __len__(self):
        return self.max_len


class SingleDiseaseDataSet(Dataset):
    """
    仅返回单个疾病的数据集的Dataset类
    """

    def __init__(self, target_diseases):
        self.target_diseases = target_diseases

        self.data_arr_dict = get_data_arr_dict()
        print("finish init data_dict")
        self.dse_df = disease_df

        # 获取疾病名称的列表
        if self.target_diseases not in disease_df.columns.to_list():
            raise ValueError('使用的疾病并未在疾病列表中')

        # 获取死亡的id
        self.death_id = death_df["idnum"].tolist()

        # 获取该疾病名称到所对应的患者id列表
        self.dse_id_list = self.dse_df[self.dse_df[self.target_diseases]]["idnum"].tolist()

    def __getitem__(self, index):
        x = self.data_arr_dict[self.dse_id_list[index]].astype('float32')
        y = 1 if self.dse_id_list[index] in self.death_id else 0
        return x, y

    def __len__(self):
        return len(self.dse_id_list)


class GlobalDiseaseDataSet(Dataset):
    """
    全局任务模型使用的数据集，不用去区分病症
    """

    def __init__(self):
        self.data_arr_dict: dict = get_data_arr_dict()

        # 获取死亡的id
        self.death_id = death_df["idnum"].tolist()
        self.keys_list = list(self.data_arr_dict.keys())

    def __getitem__(self, index):
        id_num = self.keys_list[index]

        x = self.data_arr_dict[id_num].astype('float64')
        y = 1 if id_num in self.death_id else 0
        return x, y

    def __len__(self):
        return len(self.data_arr_dict)


if __name__ == "__main__":
    save_data_arr_dict()
