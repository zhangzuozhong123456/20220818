import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.linear_model import RandomizedLasso


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

processed_data = np.load('processed_data.npy')
processed_name = np.load('processed_name.npy')
print('数据集维度：', processed_data.shape)
print('ID维度：', processed_name.shape)
scores = defaultdict(list)

'''稳定性选择是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。
它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。
理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。'''

X = preprocessing.StandardScaler().fit_transform(processed_data[:, 1:])
Y = processed_data[:, 0]

rf = RandomForestRegressor()
for train_idx, test_idx in ShuffleSplit(n_splits=20).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[processed_name[i]].append((acc - shuff_acc) / acc)
print("Features sorted by their score:")
a = sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
print(a)
top20 = []
for i in a[:20]:
    top20.append(i[1])
print(top20)

# TotalImpo = []
# for i in range(3):
#     print('第',i,'颗树训练中......')
#     IndexImp = {}
#     clf = RandomForestRegressor()
#     clf.fit(preprocessed_data, processed_data[:,0])
#     importance = clf.feature_importances_
#     TotalImpo.append(importance)
#
# importance = np.mean(np.array(TotalImpo),axis=0)
# indices = np.argsort(importance)
#
# ID = []
# for i in range(preprocessed_data.shape[1]):
#     if i in indices[-20:]:
#         ID.append(processed_name[i])
# print(ID)


font = {
    'weight': 'normal',
    'size': 16,
}
plt.plot([i for i in range(preprocessed_data.shape[1])], sorted(importance, reverse=True),
         label='基于随机森林特征选择的重要度排序\nk=4')
# plt.bar(indices[30:], importance[30:], 1, color="yellow",label = '基于F检验特征选择的重要度')
# plt.bar(indices[:30], importance[:30], 1, color="blue")
plt.xlabel('特征', font)
plt.ylabel('重要度', font)
plt.show()

# total= total+index
#
# #判断某特征出现的次数
# dict = {}
# for key in total:
#     dict[key] = dict.get(key, 0) + 1
# print(dict)
#
#
# name_ = [name[i] for i in index]
# print(index)
# print(name_)
# im = sorted(importance,reverse=True)
# print(im)
#
# plt.plot([i for i in range(preprocessed_data.shape[1])],im)
# plt.show()


# eles = []
# for i in range(1, processed_data.shape[1]):
#     if i in top20_index:
#         eles.append(processed_name[i])
# print(eles)
#
# font = {'size': 24}
# plt.bar([i for i in range(processed_data.shape[1])], meanCors, width=1, color="#87CEFA")
# plt.bar(0, meanCors[0], width=1, color="red")
# plt.tick_params(labelsize=20)
# plt.xlabel('分子描述符变量索引', font)
# plt.ylabel('与化合物生物活性水平的相关性（p=0.01）', font)
# plt.style.use('ggplot')
# plt.show()
