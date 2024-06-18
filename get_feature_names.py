"""
使用shap图的时候使用了dataloader，使进入的数据都成为了张量，损失了特征名称这一数据
这使得最后可视化的时候只有特征的序号，feature1，feature2之类的，很不直观
为了解决这一问题，需要传入特征名称的列表，本脚本就用来生成该列表
"""
import os
import numpy
import pandas
from translate import Translator

DATA_CSV_PATH = r"data/原始数据 KNN补全 Version4.csv"
CATE_CSV_PATH = r"data/数据类型.csv"

tj_data_df = pandas.read_csv(DATA_CSV_PATH, index_col=0)
cate_df = pandas.read_csv(CATE_CSV_PATH)

numerical_fields = cate_df[cate_df["type"] == "numerical"]["name"].tolist()
categorical_fields = cate_df[cate_df["type"] == "categorical"]["name"].tolist()
cate_num_dict = {i[0]: i[1] for i in cate_df[cate_df["type"] == "categorical"][["name", 'cate_num']].values}


def _get_data_arr_dict(i):
    """
    该函数是本来用来创造张量的
    """

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


def get_cate_feature_len():
    res_dict = {}
    for col in categorical_fields:
        value = tj_data_df.loc[1, col].count('.')
        res_dict.update({col: value})
    res_dict.update({'diet1': 2})
    return res_dict


def get_convert_dict():
    df = pandas.read_csv('data/变量名称对应关系3.0.csv')
    d = {}
    for i in df.values:
        if not pandas.isnull(i[1]):
            d.update({i[0]: i[1]})
        elif not pandas.isnull(i[2]):
            d.update({i[0]: i[2]})
        else:
            d.update({i[0]: None})
    return d


def check_convert_dict():
    df = pandas.read_csv('csv_file/变量名称对应关系 自整理.csv')
    for i in df.values:
        if not pandas.isnull(i[1]) and not pandas.isnull(i[2]):
            if i[1] != i[2]:
                print(i[1], i[2])


def get_feature_name_list(lang='英文'):
    if lang == '英文':
        df = pandas.read_csv('data/编号转中英字段名称.csv', encoding='gbk')
        cd = {i[0]: i[1] for i in zip(df['code'].values, df['en_name'].values)}
    else:
        df = pandas.read_csv('data/编号转中英字段名称.csv', encoding='gbk')
        cd = {i[0]: i[1] for i in zip(df['code'].values, df['name'].values)}
    cate_num = get_cate_feature_len()
    convert_list = [cd[i] for i in numerical_fields]
    for col in categorical_fields:
        col2 = cd[col]
        for i in range(int(cate_num[col])):
            if lang == '英文':
                col_name = col2 + f' type {str(i + 1)}'
            else:
                col_name = col2 + f'类型{str(i + 1)}'
            convert_list.append(col_name)
    return convert_list


def translate_zh_to_en(s):
    translator = Translator(from_lang='zh-cn', to_lang="en")
    translation = translator.translate(s)
    return translation


if __name__ == '__main__':
    pass
