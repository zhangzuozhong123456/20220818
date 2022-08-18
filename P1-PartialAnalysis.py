import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import pingouin as pg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

processed_data = np.load('processed_data-1016.npy')
processed_name = list(np.load('processed_name-1016.npy'))

processed_data = pd.DataFrame(processed_data)
processed_data.columns = processed_name

# a = pg.partial_corr(processed_data, x=processed_name[0], y=processed_name[1], covar=[i for i in processed_name[2:]])
#n         r          CI95%     p-val

var_name = processed_name[1:]
cor = []
for i in range(1,processed_data.shape[1]):
    Control = []
    for index,name in enumerate(var_name):
        if index != i-1:
            Control.append(name)
    a = pg.partial_corr(processed_data, x=processed_name[0], y=processed_name[i], covar=Control)
    cor.append(abs(float(a['r'])))
print(cor)


''''保存excel'''
import xlwt
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')

for i in range(processed_data.shape[1]-1):
    worksheet.write(i, 0, cor[i])
    if i in np.argsort(cor)[-20:]:
        worksheet.write(i, 1, cor[i])
workbook.save('data/ret/PaN_RET.xls')


