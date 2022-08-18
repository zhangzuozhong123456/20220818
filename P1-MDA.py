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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

processed_data = np.load('processed_data-1016.npy')
processed_name = np.load('processed_name-1016.npy')

X = processed_data[:, 1:]
Y = processed_data[:, 0]
processed_name = processed_name[1:]

print('数据集维度：', X.shape)
print('ID维度：', Y.shape)
scores = defaultdict(list)

'''平均精确率减少'''

rf = RandomForestRegressor()
step = 0
for train_idx, test_idx in ShuffleSplit(n_splits=300).split(X):
    print(step)
    step += 1
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
a = sorted([(abs(round(np.mean(score), 4)), feat) for feat, score in scores.items()], reverse=True)
print(a)


MDA_DATA = []
MDA_NAME = []
for i in range(20):
    MDA_DATA.append(a[i][0])
    MDA_NAME.append(a[i][1])
np.save('data/MDA_DATA.npy', np.array(MDA_DATA))
np.save('data/MDA_NAME.npy', np.array(MDA_NAME))



