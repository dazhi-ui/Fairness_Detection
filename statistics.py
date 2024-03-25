# load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from IPython import display
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random
from random import sample
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


def oneHotCatVars(df, df_cols):
    # 得到数值类型的所有列的数据。
    df_1 = df.drop(columns=df_cols, axis=1)
    # df[df_cols]获得数据类型为字符串的所有列的数据
    # pd.get_dummies进行虚拟变量
    # 讲所有结果统计合并到列名，生成多列，然后0-1进行标记。1表示相同
    df_2 = pd.get_dummies(df[df_cols])
    return (pd.concat([df_1, df_2], axis=1, join='inner'))

# vanilla classifier分类器
def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    TP = confusion.loc[1, 1]
    TN = confusion.loc[0, 0]
    FP = confusion.loc[0, 1]
    FN = confusion.loc[1, 0]

    print("TP={}, TN={}, FP={}, FN={}".format(TP,TN,FP,FN))

    out = {}
    out['ALL'] = (TP + TN + FP + FN)
    out['DP'] = (TP + FP) / (TP + TN + FP + FN)
    out['TPR'] = TP / (TP + FN)
    out['TNR'] = TN / (FP + TN)
    out['FPR'] = FP / (FP + TN)
    out['FNR'] = FN / (TP + FN)
    out['ACR'] = (TP + TN) / (TP + TN + FP + FN)
    return out

# 检测测试样本中的正常和异常的数量
def ceshi_bili(y):
    zheng = 0
    cuo = 0
    for i in y:
        if i == 1:
            zheng += 1
        else:
            cuo += 1
    print("ecr_test_x异常的个数{}， 正常的个数{}".format(zheng, cuo))
    print()

if __name__ == '__main__':
    # 获取数据
    data = pd.read_csv("datasets/kddcup.data_10_percent_corrected.txt")
    # 函数的作用是从字典对象导入数据，key是列名，value是数据
    data = pd.DataFrame(data)
    list_attack_category = [] # 存放的最后赋值的类别标签
    list_agreement = [] # 存放的各种协议的名称
    for line in data.itertuples():
        list_agreement.append(line[3].replace('.',''))
        if line[42].replace('.', '') == 'normal':
            list_attack_category.append(0)
        else:
            list_attack_category.append(1)
    list_agreement = list(set(list_agreement)) # 记录的是共有的协议的种类数
    data['type']=list_attack_category # 为原始数据添加标签

    print("list_agreement:",list_agreement)
    for i in list_agreement:
        zhengchang = 0
        yichang = 0
        for line in data.itertuples():
            if line[3] == i:
                if line[42].replace('.', '') == 'normal':
                    zhengchang += 1
                else:
                    yichang += 1
        print("协议名称：{}， 异常样本数量：{}， 正常样本数量：{}".format(i,yichang,zhengchang))
