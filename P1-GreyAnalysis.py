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
processed_name = np.load('processed_name-1016.npy')
# processed_data[:,1:] = StandardScaler().fit_transform(processed_data[:,1:])
# processed_data[:,0] = MinMaxScaler().fit_transform(processed_data[:,0].reshape(-1, 1)).reshape(1, -1)

'''灰色关联分析'''
model = GraModel(processed_data, p=0.01, standard=True)
result = model.result
meanCors = result['meanCors']['value']
top20_index = pd.Series(meanCors).sort_values(ascending=False).index[1:21]
top20_value = sorted(list(meanCors), reverse=True)[1:21]
print(top20_index)
print([round(i, 4) for i in top20_value])
eles = []
for i in range(processed_data.shape[1]):
    if i in top20_index:
        eles.append(processed_name[i])
print(eles)

'''保存数据'''
GRA_DATA = []
GRA_NAME = []
for i in range(processed_data.shape[1]):
    if i in top20_index:
        GRA_DATA.append(processed_data[:, i])
        GRA_NAME.append(processed_name[i])
np.save('data/GRA_DATA.npy', np.array(GRA_DATA))
np.save('data/GRA_NAME.npy', np.array(GRA_NAME))




''''画图1'''
mean1 = np.mean([round(i,4) for i in top20_value])
mean2 = np.mean(meanCors[1:])

font = {'size': 24}
plt.bar([i-1 for i in range(1,processed_data.shape[1])], meanCors[1:], width=1, color="#87CEFA")
for i in range(20):
    plt.bar(top20_index[i]-1, meanCors[top20_index[i]], width=1, color="red")

plt.axhline(y=mean1, ls=":", c="black")
plt.axhline(y=mean2, ls=":", c="green")
plt.tick_params(labelsize=20)
plt.xlabel('分子描述符变量索引', font)
plt.ylabel('与化合物生物活性水平的相关性（p=0.01）', font)
plt.style.use('ggplot')
plt.show()


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
