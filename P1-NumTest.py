import numpy as np
import xlrd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def test(a,b,c):
    descriptor_data = np.load('descriptor_data.npy')

    '''筛选变量(空值以及比例异常)'''
    non_num = 0
    abnormal = 0
    for i in range(729):
        data = list(descriptor_data[:, i])
        els = set(data)
        '''空值'''
        if len(els) == 1:
            non_num += 1
        elif len(els) != 1:
            repeat_num = {}
            for m in els:
                repeat_num[m] = data.count(m)
            repeat_num = sorted(repeat_num.items(), key=lambda item: item[1], reverse=True)

            '''大比例重复值'''
            if repeat_num[0][1] / 1973 > 0.9:
                abnormal += 1
            else:
                if b < len(repeat_num) and repeat_num[0][1] / 1973 > a and repeat_num[0][0] == 0.0 and repeat_num[1][1] / 1973 < c:
                    abnormal += 1

    # print('0值数量：', non_num)
    # print('异常值数量：', abnormal)
    # print('总剔除数量：', non_num+abnormal)
    # print('剩余特征数量：', 729-(non_num+abnormal))
    print('为',a,'为',b,'为',c,'时，剔除变量有',abnormal,'个，总剔除',non_num+abnormal,'个，剩余',729-(non_num+abnormal),'个变量')

test(0.1,50,0.03)
test(0.1,50,0.08)
test(0.1,50,0.03)
test(0.1,50,0.08)
test(0.2,50,0.03)
test(0.2,50,0.08)
test(0.2,99,0.03)
test(0.2,99,0.08)
