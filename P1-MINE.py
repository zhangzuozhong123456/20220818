import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


X = np.load('processed_data-1016.npy')[:,1:]
Y = np.load('processed_data-1016.npy')[:,0]
processed_name = list(np.load('processed_name-1016.npy')[1:])


'''基于最大信息系数'''
m = MINE()
importance = []
for i in range(X.shape[1]):
    m.compute_score(Y, X[:,i])
    importance.append(m.mic())

print(importance)
indices = np.argsort(importance)[-20:]

ID = []
for i in range(X.shape[1]):
    if i in indices:
        ID.append(processed_name[i])
print(ID)


'''保存数据'''
GRA_DATA = []
GRA_NAME = []

for i,id in enumerate(processed_name):
    if id in ID:
        GRA_DATA.append(X[:, i])
        GRA_NAME.append(processed_name[i])
np.save('data/MINE_DATA.npy', np.array(GRA_DATA))
np.save('data/MINE_NAME.npy', np.array(GRA_NAME))


''''保存excel'''
import xlwt
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')

for i in range(X.shape[1]):
    worksheet.write(i, 0, importance[i])
    if i in np.argsort(importance)[-20:]:
        worksheet.write(i, 1, importance[i])
workbook.save('data/MINE_RET.xls')




