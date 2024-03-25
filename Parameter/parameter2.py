# load data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from IPython import display
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
import math
import cvxpy as cp
from lightgbm import LGBMClassifier
from pyod.models.knn import KNN # 2000
from pyod.models.ecod import ECOD # 2022
from pyod.models.abod import ABOD # 2008
from pyod.models.copod import COPOD # 2020
from pyod.models.iforest import IForest # 2008
from pyod.models.loda import LODA # 2016
from pyod.models.lunar import LUNAR # 2022
from pyod.models.inne import INNE # 2018
from pyod.models.deep_svdd import DeepSVDD # 2018
from pyod.models.alad import ALAD # 2018
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import minmax_scale
from scipy.spatial import KDTree
from sklearn.preprocessing import normalize
from sklearn import preprocessing
warnings.filterwarnings("ignore")



# 讲字符串列，转换为多个0-1的列，与原有的数值型列相加
def oneHotCatVars(df, df_cols):
    # 得到数值类型的所有列的数据。
    df_1 = df.drop(columns=df_cols, axis=1)
    # df[df_cols]获得数据类型为字符串的所有列的数据
    # pd.get_dummies进行虚拟变量
    # 讲所有结果统计合并到列名，生成多列，然后0-1进行标记。1表示相同
    df_2 = pd.get_dummies(df[df_cols])
    return (pd.concat([df_1, df_2], axis=1, join='inner'))

# 对几个数据进行标准化
def normalize(data_preprocessed,columns):
    scaler = preprocessing.StandardScaler()
    # 给data_preprocessed中的前几个整形列的数据进行归一化处理 (x-x.mean)/s.std
    data_preprocessed[columns] = scaler.fit_transform(data_preprocessed[columns])
    return data_preprocessed

# 删除掉对应的列
def drop_all_agreement(a):
    b = a.drop(['type'], axis=1)
    b = b.drop(['number2_ftp'], axis=1)
    b = b.drop(['number2_telnet'], axis=1).to_numpy()
    return b


# 凸优化
def optim(loss, a, c):
    # 开始凸优化
    A = loss
    # 定义优化变量
    x = cp.Variable(loss.shape[0])
    objective = cp.Maximize(-a * cp.sum_squares(x) + cp.sum(cp.multiply(A, x)))
    # 定义约束条件
    constraints = [0 <= x, cp.sum(x) == c]
    # 定义优化问题
    prob = cp.Problem(objective, constraints)
    # 问题求解
    result = prob.solve()
    for i in range(x.value.shape[0]):
        if abs(x.value[i]) < 0.01 or x.value[i] < 0:
            x.value[i] = 0
    x.value = x.value
    return x.value # 返回优化变量的值

# 计算两次参数的误差
def dif(a, b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    sum0 = sum ** 0.5
    print()
    print(sum0 > 0.0001,end="  ")
    print("sum0:", sum0)
    return sum0


# vanilla classifier分类器
def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    try:
        TP = confusion.loc[1, 1]
    except:
        TP = 0
    try:
        TN = confusion.loc[0, 0]
    except:
        TN = 0
    try:
        FP = confusion.loc[0, 1]
    except:
        FP = 0
    try:
        FN = confusion.loc[1, 0]
    except:
        FN = 0

    print("TP={}, TN={}, FP={}, FN={}".format(TP,TN,FP,FN))

    out = {}
    out['ALL'] = (TP + TN + FP + FN)
    out['DP'] = (TP + FP) / (TP + TN + FP + FN)

    # 常见的DI.标签是1的/标签是0的
    out['DI'] = (TP / (TP + FN)) / (TN / (TN + FP))

    out['TPR'] = TP / (TP + FN)
    out['TNR'] = TN / (FP + TN)
    out['FPR'] = FP / (FP + TN)
    out['FNR'] = FN / (TP + FN)
    out['ACR'] = (TP + TN) / (TP + TN + FP + FN)
    return out


# 数据获取
def get_datasets():
    data = pd.read_csv("datasets/NSLKDDTrain+_20Percent.txt")
    data = pd.DataFrame(data)
    data = data.drop(['number42'], axis=1)
    list_attack_category = []
    list_agreement = []
    for line in data.itertuples():
        list_agreement.append(line[3].replace('.', ''))
        if line[42].replace('.', '') == 'normal':
            list_attack_category.append(0)
        else:
            list_attack_category.append(1)
    list_agreement = list(set(list_agreement))
    data['type'] = list_attack_category

    # ['ftp','eco_i','telnet','private','finger','telnet','ftp','telnet','ftp','auth']
    # 获得这些数据
    data_ftp = data[data['number2'] == 'ftp']
    data_telnet = data[data['number2'] == 'telnet']

    data = pd.concat([data_ftp, data_telnet])
    data = data.drop(['number41'], axis=1)
    for col in set(data.columns) - set(data.describe().columns):
        data[col] = data[col].astype('category')
    data_preprocessed = oneHotCatVars(data, data.select_dtypes('category').columns)
    normalize_columns = ['number0']
    for i in range(4, 41):
        name = 'number' + str(i)
        normalize_columns.append(name)
    return normalize(data_preprocessed, normalize_columns)

# 数据预预处理部分
def do_datasets(data_preprocessed):
    x_train, x_test = train_test_split(data_preprocessed)
    ftp_test = x_test[x_test['number2_ftp'] == 1]
    telnet_test = x_test[x_test['number2_telnet'] == 1]
    train = drop_all_agreement(x_train)
    train_label = x_train['type']

    test_x = drop_all_agreement(x_test)
    test_label = x_test['type']
    ftp_test_x = drop_all_agreement(ftp_test)
    ftp_test_y = ftp_test['type']
    telnet_test_x = drop_all_agreement(telnet_test)
    telnet_test_y = telnet_test['type']

    # ftp
    x_train_ftp = x_train[x_train['number2_ftp'] == 1]
    x_train_ftp_p = x_train_ftp[x_train_ftp['type'] == 1]
    x_train_ftp_n = x_train_ftp[x_train_ftp['type'] == 0]

    train_ftp_p = drop_all_agreement(x_train_ftp_p)
    ftp_p_label = x_train_ftp_p['type'].to_numpy()

    train_ftp_n = drop_all_agreement(x_train_ftp_n)
    ftp_n_label = x_train_ftp_n['type'].to_numpy()

    # telnet
    x_train_telnet = x_train[x_train['number2_telnet'] == 1]
    x_train_telnet_p = x_train_telnet[x_train_telnet['type'] == 1]
    x_train_telnet_n = x_train_telnet[x_train_telnet['type'] == 0]

    train_telnet_p = drop_all_agreement(x_train_telnet_p)
    telnet_p_label = x_train_telnet_p['type'].to_numpy()

    train_telnet_n = drop_all_agreement(x_train_telnet_n)
    telnet_n_label = x_train_telnet_n['type'].to_numpy()


    # 对训练数据进行一个组合
    train_wen = np.concatenate(
        [train_ftp_p, train_ftp_n, train_telnet_p, train_telnet_n])
    label_wen = np.concatenate(
        (ftp_p_label, ftp_n_label, telnet_p_label, telnet_n_label))

    train_wen_p = np.concatenate(
        [train_ftp_p,train_telnet_p])
    label_wen_p = np.concatenate(
        (ftp_p_label, telnet_p_label))

    return x_train_ftp_p,x_train_ftp_n, x_train_telnet_p,x_train_telnet_n,\
           train_ftp_p, train_ftp_n, train_telnet_p, train_telnet_n,\
           ftp_p_label, ftp_n_label, telnet_p_label, telnet_n_label,\
           ftp_test_x,telnet_test_x,ftp_test_y,telnet_test_y,\
           test_x,test_label,\
           x_train,\
           train_wen,label_wen,train_wen_p,label_wen_p

# 采用fairness进行测试
def fairness():
    loss_ftp_p = wi0_ftp_p = np.ones(x_train_ftp_p.shape[0])
    loss_ftp_n = wi0_ftp_n = np.ones(x_train_ftp_n.shape[0])

    loss_telnet_p = wi0_telnet_p = np.ones(x_train_telnet_p.shape[0])
    loss_telnet_n = wi0_telnet_n = np.ones(x_train_telnet_n.shape[0])

    a = 1
    c = 1000
    iter = 0
    # 用于权重的更新操作
    wi0 = np.ones((x_train.shape[0]))
    train_wen = np.concatenate(
        [train_ftp_p, train_ftp_n, train_telnet_p, train_telnet_n])
    label_wen = np.concatenate(
        (ftp_p_label, ftp_n_label, telnet_p_label, telnet_n_label))
    wi1 = wi0 + 1000000000000000000
    mm = 0
    while dif(wi0, wi1) > 0.0001 and mm<=20:
        mm += 1
        log_reg = LGBMClassifier(num_leaves=128, n_estimators=100)
        if mm == 1:
            wi0 = np.ravel(U)
        log_reg.fit(train_wen, label_wen, wi0)

        # ftp
        for i2 in range(train_ftp_n.shape[0]):
            train_ftp_n[i2] = -math.log(log_reg.predict_proba(train_ftp_n)[i2][0] + 1e-100)

        for i1 in range(train_ftp_p.shape[0]):
            loss_ftp_p[i1] = -math.log(log_reg.predict_proba(train_ftp_p)[i1][1] + 1e-100)

        # telnet
        for i2 in range(train_telnet_n.shape[0]):
            train_telnet_n[i2] = -math.log(log_reg.predict_proba(train_telnet_n)[i2][0] + 1e-100)

        for i1 in range(train_telnet_p.shape[0]):
            loss_telnet_p[i1] = -math.log(log_reg.predict_proba(train_telnet_p)[i1][1] + 1e-100)

        wi1 = wi0
        wi0_ftp_p_1 = optim(loss_ftp_p, a, c)
        wi0_ftp_n_1 = optim(loss_ftp_n, a, c)

        wi0_telnet_p_1 = optim(loss_telnet_p, a, c)
        wi0_telnet_n_1 = optim(loss_telnet_n, a, c)

        wi0 = np.concatenate((wi0_ftp_p_1, wi0_ftp_n_1, wi0_telnet_p_1,
                              wi0_telnet_n_1))
        iter = iter + 1

        log_reg_pred1 = log_reg.predict(ftp_test_x)  # 0,1标签组合
        logistic_reg1 = model_eval(ftp_test_y, log_reg_pred1)

        log_reg_pred3 = log_reg.predict(telnet_test_x)
        logistic_reg3 = model_eval(telnet_test_y, log_reg_pred3)

        log_reg_pred11 = log_reg.predict(test_x)
        logistic_reg11 = model_eval(test_label, log_reg_pred11)

        list_linshi_di = [logistic_reg1['DP'], logistic_reg3['DP']]
        mean_di = np.mean(list_linshi_di)
        DI = round(100 * abs(
            abs(logistic_reg1['DP'] - mean_di) + abs(logistic_reg3['DP'] - mean_di)), 4)
        list_linshi_TNR = [logistic_reg1['TNR'], logistic_reg3['TNR']]
        mean_TNR = np.mean(list_linshi_TNR)
        DFPR = round(100 * abs(abs(logistic_reg1['TNR'] - mean_TNR) + abs(logistic_reg3['TNR'] - mean_TNR)), 4)
        list_linshi_TPR = [logistic_reg1['TPR'], logistic_reg3['TPR']]
        mean_TPR = np.mean(list_linshi_TPR)
        DFNR = round(100 * abs(abs(logistic_reg1['TPR'] - mean_TPR) + abs(logistic_reg3['TPR'] - mean_TPR)), 4)

        prule = round(
            100 * min(logistic_reg1['DP'] / logistic_reg3['DP'],
                      logistic_reg3['DP'] / logistic_reg1['DP']), 4)
        ACR = round(logistic_reg11['ACR'] * 100, 4)
        list_accuracy.append(ACR)
        list_DI.append(DI)
        list_FPR.append(DFPR)
        list_FNR.append(DFNR)
        list_disparate.append(DI + DFPR + DFNR)
        list_rule.append(prule)
        print("这是第 {} 个".format(mm))
        print(
            f'Our method: Iteration {iter - 1}, Average accuracy is {ACR}% , disparate impact is {DI}% , disparate FPR is {DFPR}% , disparate FNR is {DFNR}% , p% rule is {prule}% ，')

    print("Average数值最大的：")
    value = max(list_accuracy)
    list_accuracy_disparate = []
    keys_1 = []
    for i in range(0, len(list_accuracy)):
        if list_accuracy[i] == value:
            list_accuracy_disparate.append(list_disparate[i])
            keys_1.append(i)
    value = min(list_accuracy_disparate)
    for i in range(0, len(list_accuracy_disparate)):
        if list_accuracy_disparate[i] == value:
            i2 = keys_1[i]
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i2]}% , disparate impact is {list_DI[i2]}% , disparate FPR is {list_FPR[i2]}% , disparate FNR is {list_FNR[i2]}% , p% rule is {list_rule[i2]}% ，')

    print()
    value = min(list_disparate)
    list_disparate_accuracy = []
    keys_2 = []
    print("disparate数值最小的：")
    for i in range(0, len(list_disparate)):
        if list_disparate[i] == value:
            list_disparate_accuracy.append(list_accuracy[i])
            keys_2.append(i)
    value = max(list_disparate_accuracy)
    for i in range(0, len(list_disparate_accuracy)):
        if list_disparate_accuracy[i] == value:
            i2 = keys_2[i]
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i2]}% , disparate impact is {list_DI[i2]}% , disparate FPR is {list_FPR[i2]}% , disparate FNR is {list_FNR[i2]}% , p% rule is {list_rule[i2]}% ，')

    print()
    print("DI数值最小的：")
    value = min(list_DI)
    for i in range(0, len(list_DI)):
        if list_DI[i] == value:
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i]}% , disparate impact is {list_DI[i]}% , disparate FPR is {list_FPR[i]}% , disparate FNR is {list_FNR[i]}% , p% rule is {list_rule[i]}% ，')
    print()
    print("FPR数值最小的：")
    value = min(list_FPR)
    for i in range(0, len(list_FPR)):
        if list_FPR[i] == value:
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i]}% , disparate impact is {list_DI[i]}% , disparate FPR is {list_FPR[i]}% , disparate FNR is {list_FNR[i]}% , p% rule is {list_rule[i]}% ，')
    print()
    print("FNR数值最小的：")
    value = min(list_FNR)
    for i in range(0, len(list_FNR)):
        if list_FNR[i] == value:
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i]}% , disparate impact is {list_DI[i]}% , disparate FPR is {list_FPR[i]}% , disparate FNR is {list_FNR[i]}% , p% rule is {list_rule[i]}% ，')
    print()
    print("rule数值最大的：")
    value = max(list_rule)
    for i in range(0, len(list_rule)):
        if list_rule[i] == value:
            print(
                f'Average数值最大的: Average accuracy is {list_accuracy[i]}% , disparate impact is {list_DI[i]}% , disparate FPR is {list_FPR[i]}% , disparate FNR is {list_FNR[i]}% , p% rule is {list_rule[i]}% ，')

    return train_wen,label_wen

def diyibufen_method(log_reg,name,label,train):
    log_reg.fit(train, label)

    log_reg_pred1 = log_reg.predict(ftp_test_x)  # 0,1标签组合ftps
    logistic_reg1 = model_eval(ftp_test_y, log_reg_pred1)

    log_reg_pred3 = log_reg.predict(telnet_test_x)
    logistic_reg3 = model_eval(telnet_test_y, log_reg_pred3)

    log_reg_pred11 = log_reg.predict(test_x)
    logistic_reg11 = model_eval(test_label, log_reg_pred11)

    list_linshi_di = [logistic_reg1['DP'] , logistic_reg3['DP'] ]
    mean_di = np.mean(list_linshi_di)
    DI = round(100 * abs(
        abs(logistic_reg1['DP']-mean_di)+abs(logistic_reg3['DP']-mean_di)), 4)
    list_linshi_TNR = [logistic_reg1['TNR'],  logistic_reg3['TNR']]
    mean_TNR = np.mean(list_linshi_TNR)
    DFPR = round(100 * abs(abs(logistic_reg1['TNR']-mean_TNR)+abs(logistic_reg3['TNR']-mean_TNR)), 4)
    list_linshi_TPR = [logistic_reg1['TPR'], logistic_reg3['TPR']]
    mean_TPR = np.mean(list_linshi_TPR)
    DFNR = round(100 * abs(abs(logistic_reg1['TPR']-mean_TPR)+abs(logistic_reg3['TPR']-mean_TPR)), 4)

    prule = round(100 * min(logistic_reg1['DP'] / logistic_reg3['DP'],

                            logistic_reg3['DP'] / logistic_reg1['DP']), 4)
    ACR = round(logistic_reg11['ACR'] * 100, 4)

    print(
        f'Baseline method:  Average accuracy is {ACR}%, disparate impact is {DI}%, disparate FPR is {DFPR}%, disparate FNR is {DFNR}%, p% rule is {prule}%，')

# 不使用fairness进行测试
def unfairness():
    train = train_wen
    train_label = label_wen

    log_reg = LGBMClassifier(num_leaves=128, n_estimators=100)
    name = "LGBMClassifier"
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    log_reg = LogisticRegression(penalty='l2', dual=False, tol=1e-4, fit_intercept=False,
                                 max_iter=400, solver='newton-cg', warm_start=True)
    name = "LogisticRegression"
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    log_reg = KNeighborsClassifier()
    name = "KNeighborsClassifier"
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "KNN"
    log_reg = KNN()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "ECOD"
    log_reg = ECOD()
    label = None
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "ABOD"
    log_reg = ABOD()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "COPOD"
    log_reg = COPOD()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "IForest"
    log_reg = IForest()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "LODA"
    log_reg = LODA()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "LUNAR"
    log_reg = LUNAR()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "INNE"
    log_reg = INNE()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "ALAD"
    log_reg = ALAD()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

    name = "DeepSVDD"
    log_reg = DeepSVDD()
    label = train_label
    diyibufen_method(log_reg, name, label, train)
    print()

# 用于给异常数据聚类使用
def cluster(anomaly, k):
    k_means = KMeans(n_clusters=k)
    result = k_means.fit_predict(anomaly)
    return k_means.cluster_centers_, result

# 初始顶点权重
def init_vertex_weight(train_wen, train_wen_p, centers, eta):
    num = train_wen.shape[0]
    anomaly_num = int(train_wen_p.shape[0])
    TS = calculate_score(train_wen, centers, eta)
    gamma = np.mean(TS[:anomaly_num])
    U = np.zeros((num, 1))
    abnormal_index = np.nonzero(TS > gamma)[0]
    normal_index = np.nonzero(TS <= gamma)[0]
    U[abnormal_index] = TS[abnormal_index] / np.max(TS)
    U[normal_index] = (np.max(TS) - TS[normal_index]) / (np.max(TS) - np.min(TS))
    U[:anomaly_num] = 1
    return U

def e_distances(A, B):
    BT = B.transpose()
    vec_prod = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vec_prod.shape[1]))
    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vec_prod.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vec_prod
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

# 计算分数
def calculate_score(test_data, centers, eta):
    clf = IsolationForest(n_estimators=100, contamination=0.03, n_jobs=-1)
    clf.fit(test_data)
    isolation_score = -clf.decision_function(test_data)
    num = test_data.shape[0]
    IS = minmax_scale(isolation_score).reshape((num, 1))
    distance = np.array(np.min(e_distances(test_data, centers), axis=1))
    dis_min = np.min(distance)
    dis_max = np.max(distance)
    distance = (distance - dis_min) / (dis_max - dis_min)
    print('distance\n', distance)
    SS = np.exp(-distance).reshape((num, 1))
    SS = minmax_scale(SS)
    TS = eta*IS + (1-eta)*SS
    print('IS\n', IS.reshape(1, -1))
    print('SS\n', SS.reshape(1, -1))
    print('TS\n', TS.reshape(1, -1))
    return TS

# 计算最终的anomaly_score，也是权重
def anomaly_score():
    # ETA是占比TS = eta*IS + (1-eta)*SS
    # 对异常的数据进行聚类操作。result是所属的簇类号
    centers, result = cluster(train_wen_p, k_cluster)
    return init_vertex_weight(train_wen, train_wen_p, centers, ETA)


# 执行程序的主函数
if __name__ == '__main__':

    #计算异常得分用到的
    k_cluster = 20
    ETA = 0.8
    # 相似性和孤立性的占比

    list_accuracy = []
    list_DI = []
    list_DI_2 = []
    list_FPR = []
    list_FNR = []
    list_disparate = []
    list_rule = []

    data_preprocessed = get_datasets()
    x_train_ftp_p, x_train_ftp_n, x_train_telnet_p, x_train_telnet_n, \
    train_ftp_p, train_ftp_n, train_telnet_p, train_telnet_n, \
    ftp_p_label, ftp_n_label, telnet_p_label, telnet_n_label, \
    ftp_test_x, telnet_test_x, ftp_test_y, telnet_test_y, \
    test_x, test_label, \
    x_train, \
    train_wen, label_wen, train_wen_p, label_wen_p = do_datasets(data_preprocessed)

    U = anomaly_score()
    fairness()

    unfairness()


