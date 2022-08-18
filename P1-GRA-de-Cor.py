import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GraModel():
    '''灰色关联度分析模型'''

    def __init__(self, inputData, p=0.5, standard=True):
        '''
        初始化参数
        inputData：输入矩阵，纵轴为属性名，第一列为母序列
        p：分辨系数，范围0~1，一般取0.5，越小，关联系数间差异越大，区分能力越强
        standard：是否需要标准化
        '''
        self.inputData = np.array(inputData)
        self.p = p
        self.standard = standard
        # 标准化
        self.standarOpt()
        # 建模
        self.buildModel()

    def standarOpt(self):
        '''标准化输入数据'''
        if not self.standard:
            return None
        self.scaler = StandardScaler().fit(self.inputData)
        self.inputData = self.scaler.transform(self.inputData)

    def buildModel(self):
        # 第一列为母列，与其他列求绝对差
        momCol = self.inputData[:, 0]
        sonCol = self.inputData[:, 0:]
        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col] - momCol)
        # 求两级最小差和最大差
        minMin = sonCol.min()
        maxMax = sonCol.max()
        # 计算关联系数矩阵
        cors = (minMin + self.p * maxMax) / (sonCol + self.p * maxMax)
        # 求平均综合关联度
        meanCors = cors.mean(axis=0)
        self.result = {'cors': {'value': cors, 'desc': '关联系数矩阵'}, 'meanCors': {'value': meanCors, 'desc': '平均综合关联系数'}}


processed_data = np.load('processed_data-1016.npy')

X = processed_data[:, 1:]
Y = processed_data[:, 0]
processed_name = np.load('processed_name-1016.npy')[1:]

'''灰色关联分析'''
model = GraModel(processed_data, p=0.01, standard=True)
result = model.result
meanCors = result['meanCors']['value']

TOP = 80
top20_index = [i - 1 for i in pd.Series(meanCors).sort_values(ascending=False).index[1:TOP + 1]]
top20_value = sorted(list(meanCors), reverse=True)[1:TOP + 1]

# ID = []
# for i in range(X.shape[1]):
#     if i in top20_index:
#         ID.append(processed_name[i])
# print(ID)

DATA = []
NAME = []
for i in top20_index:
    DATA.append(X[:, i])
    NAME.append(processed_name[i])
DATA = np.transpose(DATA, (1, 0))
print(NAME)

df = pd.DataFrame(DATA)
df.columns = NAME
dfData = df.corr().abs()

name = []
name.append(NAME[0])
for i in range(TOP):
    ind = 0
    for j in range(i):
        if dfData[NAME[i]][j] > 0.6 and i != j:
            ind += 1
    if ind == 0:
        name.append(NAME[i])
name.pop(0)
print(len(name))


new_DATA = []
new_NAME = []
for i, id in enumerate(processed_name):
    if id in name:
        new_DATA.append(X[:, i])
        new_NAME.append(processed_name[i])
np.save('data/GRA_deCor_DATA.npy', np.array(new_DATA))
np.save('data/GRA_deCor_NAME.npy', np.array(new_NAME))


''''保存excel'''
# import xlwt
# workbook = xlwt.Workbook(encoding = 'utf-8')
# worksheet = workbook.add_sheet('My Worksheet')
#
# for i in range(1,len(list(meanCors))):
#     worksheet.write(i-1, 0, meanCors[i])
#     if i in top20_index:
#         worksheet.write(i-1, 1, meanCors[i])
# workbook.save('data/GRA_RET.xls')
