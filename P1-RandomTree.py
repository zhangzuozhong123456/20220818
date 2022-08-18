import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


X = np.load('processed_data-1016.npy')[:,1:]
Y = np.load('processed_data-1016.npy')[:,0]
processed_name = np.load('processed_name-1016.npy')[1:]
print('数据集维度：',X.shape)
print('ID维度：',processed_name.shape)


TotalImpo = []
for i in range(100):
    print('第',i,'颗树训练中......')
    IndexImp = {}
    clf = RandomForestRegressor()
    clf.fit(X, Y)
    importance = clf.feature_importances_
    TotalImpo.append(importance)

importance = np.mean(np.array(TotalImpo),axis=0)
indices = np.argsort(importance)

ID = []
for i in range(X.shape[1]):
    if i in indices[-20:]:
        ID.append(processed_name[i])



ii = 'RT'
DATA = []
NAME = []
for i in range(X.shape[1]):
    if i in indices[-20:]:
        DATA.append(X[:,i])
        NAME.append(processed_name[i])
np.save('data/'+ii+'_DATA.npy', np.array(DATA))
np.save('data/'+ii+'_NAME.npy', np.array(NAME))

''''保存excel'''
import xlwt
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')

for i in range(X.shape[1]):
    worksheet.write(i, 0, importance[i])
    if i in indices[-20:]:
        worksheet.write(i, 1, importance[i])
workbook.save('data/'+ii+'_RET.xls')




# font = {
#          'weight': 'normal',
#          'size': 16,
#          }
# plt.plot([i for i in range(preprocessed_data.shape[1])],sorted(importance,reverse=True),label = '基于随机森林特征选择的重要度排序\nk=4')
# # plt.bar(indices[30:], importance[30:], 1, color="yellow",label = '基于F检验特征选择的重要度')
# # plt.bar(indices[:30], importance[:30], 1, color="blue")
# plt.xlabel('特征', font)
# plt.ylabel('重要度', font)
# plt.show()




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
