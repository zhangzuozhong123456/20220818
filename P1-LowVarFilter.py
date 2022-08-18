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


Molecular_Descriptor = r'Molecular_Descriptor.xlsx'
ER_activity = r'ER_activity.xlsx'
Molecular_Descriptor = xlrd.open_workbook(Molecular_Descriptor)
table_Molecular_Descriptor = Molecular_Descriptor.sheet_by_name('training')
ER_activity = xlrd.open_workbook(ER_activity)
table_ER_activity = ER_activity.sheet_by_name('training')

IC50 = table_ER_activity.col_values(1)[1:]
pIC50 = table_ER_activity.col_values(2)[1:]
name = table_Molecular_Descriptor.row_values(0)[1:]

descriptor_data = []
for i in range(1974):
    data = table_Molecular_Descriptor.row_values(i + 1)[1:]
    data.insert(0, IC50[i])
    descriptor_data.append(data)
descriptor_data = np.array(descriptor_data)  # 1974 x 730

'''筛选变量(空值以及比例异常)'''
exculde = []  # 225
exculde_name = []  # 225
repeat_ratio = []
non_num = 0
abnormal = 0
for i in range(730):
    data = list(descriptor_data[:, i])
    els = set(data)
    if len(els) == 1:
        non_num += 1
        exculde.append(i)
        exculde_name.append(name[i - 1])
    if len(els) != 1:
        repeat_num = {}
        for m in els:
            repeat_num[m] = data.count(m)
        repeat_num = sorted(repeat_num.items(), key=lambda item: item[1], reverse=True)
        if repeat_num[0][1] / 1974 > 0.9:
            abnormal += 1
            exculde.append(i)
            exculde_name.append(name[i - 1])
print('0值数量：', non_num)
print('异常值数量：', abnormal)
print('总剔除数量：', len(exculde))

'''计算处理后的ID的名字'''
processed_name = []
for j in range(729):
    if j + 1 not in exculde:
        processed_name.append(name[j])
print('删选后ID数量：', len(processed_name))

'''计算删选后数据'''
processed_data = []
for i in range(1974):
    ele = []
    for j in range(730):
        if j not in exculde:
            ele.append(descriptor_data[i, j])
    processed_data.append(ele)
processed_data = np.array(processed_data)
print('处理后数据：', processed_data.shape)

var_total = []
for i in range(1,processed_data.shape[1]):
    var = np.var(processed_data[:,i])
    var_total.append(var)

indices = np.argsort(var_total)

ID = []
for i in range(processed_data.shape[1]):
    if i in indices[-20:]:
        ID.append(processed_name[i])
print(ID)
print([round(i,4) for i in sorted(var_total)])





# ID = []
# for i in range(preprocessed_data.shape[1]):
#     if i in indices[:20]:
#         ID.append(processed_name[i])
# print(ID)





