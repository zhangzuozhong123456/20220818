import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest,f_classif,f_regression,mutual_info_regression


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

processed_data = np.load('processed_data.npy')
processed_name = np.load('processed_name.npy')
print('数据集维度：',processed_data.shape)
print('ID维度：',processed_name.shape)


preprocessed_data = preprocessing.StandardScaler().fit_transform(processed_data[:,1:])

dest_func = [mutual_info_regression,f_regression]
feature_select = SelectKBest(score_func=dest_func[0],k=20)
feature_select.fit(preprocessed_data,processed_data[:,0])
new_data = feature_select.transform(preprocessed_data)

index = feature_select.get_support()
scores = feature_select.scores_
pvalues = feature_select.pvalues_

selected_index = []
dropped_index = []
selected_score = []
dropped_score = []
for i in range(preprocessed_data.shape[1]):
    if int(index[i]) != 0:
        selected_index.append(i)
        selected_score.append(scores[i])
    else:
        dropped_index.append(i)
        dropped_score.append(scores[i])

ID = []
for i in range(preprocessed_data.shape[1]):
    if i in selected_index:
        ID.append(processed_name[i])
print(ID)



plt.bar(dropped_index, dropped_score, 1, color="yellow",label = '基于F检验特征选择的重要度')
plt.bar(selected_index, selected_score, 1, color="blue")
font = {
         'weight': 'normal',
         'size': 16,
         }
plt.legend(prop=font)
plt.xlabel('特征', font)
plt.ylabel('重要度', font)

plt.show()

