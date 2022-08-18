import pandas as pd
import numpy as np
import xlrd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,f_classif,f_regression,mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from collections import Counter
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from itertools import cycle

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
descriptor_data = []

for i in range(1974):
    data = table_Molecular_Descriptor.row_values(i+1)[1:]
    descriptor_data.append(data)
descriptor_data = np.array(descriptor_data)
IC50 = np.array(IC50)

# scale = preprocessing.StandardScaler().fit(data)
# data = scale.transform(data)

def featureSelect():
    preprocessed_data,preprocessed_name = preprocess()

    scale = preprocessing.StandardScaler().fit(preprocessed_data)
    preprocessed_data = scale.transform(preprocessed_data)
    '''
        特征选择

    ['氢油比', '反应器顶底压差', 'D105流化氢气流量', '稳定塔液位', '塔顶回流罐D201液位', '0.3MPa凝结水出装置流量',
    '1#催化汽油进装置流量', '还原器温度', 'D-125液位', 'D-105下锥体松动风流量', 'D-102温度', 'A-202A/B出口总管温度',
    '由PDI2104计算出', 'R-101床层中部温度', '1.1步骤PIC2401B.OP', '3.0步骤PIC2401A.SP', 'K-102A进气压力', 'EH-102加热元件/A束温度',
     'EH-102出口空气总管温度', '预热器出口空气温度', '对流室出口温度', 'E-101ABC管程出口温度', 'E-101DEF管程出口温度', 'E-101ABC壳程出口温度',
      'E-101DEF壳程出口温度', 'E-101壳程出口总管温度', '反应器藏量', '反应器质量空速', '8.0MPa氢气至循环氢压缩机入口', '8.0MPa氢气至反吹氢压缩机出口']
    [13, 20, 25, 29, 30, 49, 58, 88, 143, 175, 178, 185, 201, 250, 255, 256, 284, 311, 312, 340, 341, 346, 347, 349, 350, 351, 353, 354, 363, 365]
    '''

    '''
    树：
    ['辛烷值RON', 'D101原料缓冲罐压力', 'K-103A进气温度', '精制汽油出装置温度', 'R-101下部床层压降', 'P-101B入口过滤器差压', 
    '稳定塔下部温度', 'D-204液位', '原料换热器管程进出口压差', '蒸汽进装置流量', '加热炉主火嘴瓦斯入口压力', '精制汽油出装置硫含量', 
    'E-101D壳程出口管温度', '硫含量,μg/g', 'D121去稳定塔流量', 'R-102底喷头压差', '还原器流化氢气流量', '还原器温度', '燃料气进装置流量', 
    '氢油比', 'F-101辐射室出口压力', 'D-123压力', 'D-102温度', 'D-125液位', '饱和烃,v%（烷烃+环烷烃）', 'S_ZORB AT-0009', 'ME-105过滤器压差', 
    '过滤器ME-101出口温度', 'ME-109过滤器差压', '稳定塔顶压力']
    [8, 366, 277, 32, 245, 258, 28, 135, 68, 36, 76, 34, 132, 7, 85, 238, 16, 88, 43, 13, 300, 149, 178, 143, 2, 332, 264, 337, 262, 27]
    '''
    dest_func = [mutual_info_regression,f_regression]
    feature_select = SelectKBest(score_func=dest_func[1],k=30)
    feature_select.fit(preprocessed_data,label)
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

    font1 = {
             'weight': 'normal',
             'size': 16,
             }

    # print([name[i] for i in selected_index])
    print(selected_index)
    plt.bar(dropped_index, dropped_score, 1, color="yellow",label = '基于F检验特征选择的重要度')
    plt.bar(selected_index, selected_score, 1, color="blue")

    plt.legend(prop=font1)
    plt.xlabel('特征', font1)
    plt.ylabel('重要度', font1)

    plt.show()

def calcaltePerson():
    '''
    相关性图
    '''
    # dest_func = [mutual_info_regression,f_regression]
    # feature_select = SelectKBest(score_func=dest_func[0],k=30)
    # feature_select.fit(data,label)
    # new_data = feature_select.transform(data)
    index = [149,122,142,129,93,161,92,67,135,111,35,34,43,128,169,172,6,146,114,76,120,131]

    preprocessed_data,preprocessed_name = preprocess()

    scale = preprocessing.StandardScaler().fit(preprocessed_data)
    preprocessed_data = scale.transform(preprocessed_data)


    df = pd.DataFrame(preprocessed_data[:,index])

    font1 = {
             'weight': 'normal',
             'size': 20,
             }


    dfData = df.corr()
    # plt.subplots(figsize=(8, 8))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.xlabel('最终选择的22维特征相关性', font1)
    # # plt.savefig('经过第3步筛选的'+str(new_data_3.shape[1])+'维特征相关性'+".png")
    # plt.show()

    with sns.axes_style("white"):
        sns.heatmap(dfData,
                    cmap="YlGnBu",
                    annot=True,
                    )
        plt.xlabel('最终选择的22维特征相关性', font1)

    plt.show()

def _Pca():
    '''
    PCA
    '''
    pca = PCA(n_components=30)
    pca.fit(data)
    new_data = pca.fit_transform(data)

    var = pca.explained_variance_
    contri = pca.explained_variance_ratio_
    project_matrix = pca.components_
    print(var)
    print(np.sum(contri))

def eigValPct():

    pca = PCA(n_components=120)
    pca.fit(data)

    var = pca.explained_variance_
    contri = pca.explained_variance_ratio_

    sortArray = np.sort(var)

    sortArray = sortArray[::-1]

    arraySum = sum(sortArray)

    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * 1:
            print(num)
            return num

def calComponets():
    data_ = [0.32221843814109224, 0.43080479940888877, 0.49682366320168436, 0.5535831575304639, 0.5937408107044844, 0.6279689982027417, 0.6550481261670018,
     0.6785971598891466, 0.6995336208634053, 0.7187677665032156, 0.7356369132910763, 0.7499431632228185, 0.7637700988371317, 0.776049384126785,
     0.7877370601869453, 0.7978891074843101, 0.8067849884937061, 0.815115396156854, 0.8230805440178672, 0.830381332878151, 0.8372011586950217,
     0.8435271159279051, 0.849088934447362, 0.8543672364317946, 0.859611054004392, 0.8645752225174903, 0.8689683226274874, 0.8733394475928874,
     0.8775526075394229, 0.881490638241728]
    label_ = [ i for i in range(30)]

    # data_ = []
    # for i in range(1,31):
    #     print(i)
    #     pca = PCA(n_components=i)
    #     pca.fit(data)
    #
    #     contri = pca.explained_variance_ratio_
    #     print(np.sum(contri))
    #     data_.append(np.sum(contri))
    # print(data_)

    plt.plot(label_,data_)
    plt.show()

def Treebased():
    preprocessed_data,preprocessed_name = preprocess()

    scale = preprocessing.StandardScaler().fit(preprocessed_data)
    preprocessed_data = scale.transform(preprocessed_data)

    # total = []
    # for i in range(30):
    #     print(i)
    #     clf = RandomForestRegressor()
    #     clf.fit(preprocessed_data, label)
    #     importance = clf.feature_importances_
    #     total.append(importance)
    #     print(np.argsort(importance))
    #
    # importance = np.mean(np.array(total),axis=0)
    # indices = np.argsort(importance)
    # print(indices)


    clf = RandomForestRegressor()
    clf.fit(preprocessed_data, label)
    importance = clf.feature_importances_
    indices = np.argsort(importance)


    plt.plot([i for i in range(len(importance))],sorted(importance,reverse=True),label = '基于随机森林特征选择的重要度排序\nk=4')


    # plt.bar(indices[30:], importance[30:], 1, color="yellow",label = '基于F检验特征选择的重要度')
    # plt.bar(indices[:30], importance[:30], 1, color="blue")
    #
    font1 = {
             'weight': 'normal',
             'size': 16,
             }
    #
    #
    plt.legend(prop=font1)
    plt.xlabel('特征', font1)
    plt.ylabel('重要度', font1)

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

def preprocess():
    '''
    删选0多的变量 1% 342  5% 346， 10%为基准348,20%为350，30% 351 40% 354   用其他数据的均值代替
    '''
    new_data_1 = []
    new_name_1 = []
    for i in range(data.shape[1]):
        if sum(data[:,i]==0)<325*0.05:
            new_name_1.append(name[i])
            new_data_1.append(data[:,i])
        # else:
            # print(name[i])
    new_data_1 = np.transpose(new_data_1,(1,0))
    # print(new_data_1.shape[1])


    for i in range(new_data_1.shape[1]):
        none_zero_sample_index = new_data_1[:,i].nonzero()
        _mean = np.mean(new_data_1[list(none_zero_sample_index),i])

        for j in range(new_data_1.shape[0]):
            if new_data_1[j,i] == 0:
                new_data_1[j, i] = _mean

    # for i in range(new_data_1.shape[1]):
    #         if sum(new_data_1[:,i]==0)>1:
    #             print('error')
    # print('end')
    # print(data.shape[1]-new_data_1.shape[1])
    # df = pd.DataFrame(new_data_1)
    # dfData = df.corr()
    # dfData = np.array(dfData)
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }
    # plt.subplots(figsize=(8, 8))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.xlabel('经过第1步筛选的'+str(new_data_1.shape[1])+'维特征相关性', font1)
    # plt.savefig('经过第1步筛选的'+str(new_data_1.shape[1])+'维特征相关性'+".png")
    # plt.show()

    '''
    删选重复元素过多的变量 30%为基准292 20%为基准290 15% 298  10%为基准273 5%为基准268
    '''

    new_data_2 = []
    new_name_2 = []
    for i in range(new_data_1.shape[1]):
        _data = set(new_data_1[:,i])
        if len(_data)>325*0.9 or (new_name_1[i]  in ['硫含量,μg/g' , '辛烷值RON' , '饱和烃,v%（烷烃+环烷烃）' , '烯烃,v%' , '芳烃,v%' , '溴值\n,gBr/100g','密度(20℃),\nkg/m³',
                                                         'kg/m³','硫含量,μg/g','焦炭,wt%','S, wt%']):
            new_name_2.append(new_name_1[i])
            new_data_2.append(new_data_1[:,i])
        # else:
        #     print(new_name_1[i])
    new_data_2 = np.transpose(new_data_2,(1,0))
    # print('end')
    # print(new_data_1.shape[1]-new_data_2.shape[1])

    # df = pd.DataFrame(new_data_2)
    # dfData = df.corr()
    # dfData = np.array(dfData)
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }
    # plt.subplots(figsize=(8, 8))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.xlabel('经过第2步筛选的'+str(new_data_2.shape[1])+'维特征相关性', font1)
    # plt.savefig('经过第2步筛选的'+str(new_data_2.shape[1])+'维特征相关性'+".png")
    # plt.show()


    '''
    删选差异过大的变量 1e-5基准为 -3
    '''
    new_data_3 = []
    new_name_3 = []
    for i in range(new_data_2.shape[1]):
        # var = np.var(new_data_2[:,i])
        # if var>1e8:
        #     print(var)
        #     print(new_data_2[:,i])

        _max = np.max(new_data_2[:,i])
        _min = np.min(new_data_2[:,i])
        if abs((_min+1e-10)/_max) > 1e-5:
            new_data_3.append(new_data_2[:,i])
            new_name_3.append(new_name_2[i])
        # else:
        #     print(new_name_2[i])
            # print(var)
            # print(new_data_2[:,i])
            # print(standardization(new_data_2[:,i]))
    new_data_3 = np.transpose(new_data_3,(1,0))
    # print('end')
    # print(new_data_2.shape[1]-new_data_3.shape[1])




    # for i in range(74,90):
    #     print(np.sqrt(np.sum(np.square(dfData[75] - dfData[i]))))
    #
    # ditt = []
    # for i in range(74,93):
    #     p_value = dfData[i]
    #     for j in range(72,93):
    #         if i != j :
    #             to_be_compared = dfData[j]
    #             dis = np.sqrt(np.sum(np.square(p_value - to_be_compared)))
    #             ditt.append(dis)
    #             if dis>5:
    #                 print(i,j)
    #
    # # print(set(ditt))

    df = pd.DataFrame(new_data_3)
    dfData = df.corr()
    dfData = np.array(dfData)
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }
    # plt.subplots(figsize=(8, 8))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.xlabel('经过第3步筛选的'+str(new_data_3.shape[1])+'维特征相关性', font1)
    # plt.savefig('经过第3步筛选的'+str(new_data_3.shape[1])+'维特征相关性'+".png")
    # plt.show()

    # quxiang guan
    _index = []
    _dis_data = []
    for i in range(dfData.shape[0]):
        p_value = dfData[i]
        for j in range(dfData.shape[0]):
            if i!=j:
                to_be_compared = dfData[j]
                dis = np.sqrt(np.sum(np.square(p_value - to_be_compared)))
                _dis_data.append(dis)
                if dis < 1:
                    _index.append(j)
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }
    # plt.bar([i for i in range(len(set(_dis_data)))], sorted(set(_dis_data),reverse=True), label="经过第3步筛选的特征间相关性距离分布", color="#87CEFA")
    # plt.xlabel('两两特征', font1)
    # plt.ylabel('特征间相关性距离', font1)
    # plt.legend(prop=font1)
    # plt.show()




    new_data_4 = []
    new_name_4 = []
    for i in range(new_data_3.shape[1]):
        if i not in set(_index) or new_name_3[i] == '辛烷值RON':
            new_data_4.append(new_data_3[:,i])
            new_name_4.append(new_name_3[i])
        # else:
        #     print(new_name_3[i])
    new_data_4 = np.transpose(new_data_4,(1,0))
    # df = pd.DataFrame(new_data_3)
    # dfData = df.corr()
    # dfData = np.array(dfData)
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }
    # plt.subplots(figsize=(8, 8))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.xlabel('经过第4步筛选的'+str(new_data_4.shape[1])+'维特征相关性', font1)
    # plt.savefig('经过第4步筛选的'+str(new_data_4.shape[1])+'维特征相关性'+".png")
    # plt.show()

    # print('end')
    # print(new_data_3.shape[1]-new_data_4.shape[1])


    # df = pd.DataFrame(new_data_4)
    # dfData = df.corr()
    # plt.subplots(figsize=(9, 9))  # 设置画面大小
    # sns.heatmap(dfData, vmax=1, square=True, cmap="Blues")
    # plt.show()




    '''计算箱线图'''
    # new_data_4 = preprocessing.MinMaxScaler().fit_transform(new_data_4)
    # var_data = []
    # for i in range(new_data_4.shape[1]):
    #     var = np.var(new_data_4[:,i])
    #     var_data.append(var)
    #
    #
    #
    # font1 = {
    #          'weight': 'normal',
    #          'size': 16,
    #          }

    # plt.plot([i for i in range(1, 24)], total, label='全局调整', marker="o", markersize=3)
    # plt.legend(prop=font1)
    # plt.xlabel('标准化特征方差', font1)
    # plt.ylabel('辛烷值损失', font1)

    # plt.boxplot(list(set(var_data)),vert = False,sym = '+',patch_artist=True,showmeans=True)

    # plt.show()

    # var_data = sorted(var_data)
    # percentile = np.percentile(list(set(_dis_data)), (25, 50, 75), interpolation='linear')
    # Q1 = percentile[0]
    # Q3 = percentile[2]
    # IQR = Q3 - Q1
    # ulim = Q3 + 1.5 * IQR
    # llim = Q1 - 1.5 * IQR
    # print(ulim)
    # print(llim)

    return new_data_4,new_name_4

def samplediscard():
    new_data_4, new_name_4 = preprocess()
    scale = preprocessing.StandardScaler().fit(new_data_4)
    data = scale.transform(new_data_4)

    pca = PCA(n_components=2)
    pca.fit(data)
    new_data = pca.fit_transform(data)



    # kmeans = KMeans(n_clusters=4,max_iter=3000).fit(new_data)
    # k_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis=0)
    # k_means_labels = pairwise_distances_argmin(new_data, k_means_cluster_centers)
    # fig = plt.figure(figsize=(8, 3))
    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    # colors = ['#4EACC5', '#FF9C34', '#4E9A06','#4F4F4F'] #CF9904
    # ax = fig.add_subplot(1, 2, 1)
    # row, _ = np.shape(new_data)
    # for i in range(row):
    #     ax.plot(new_data[i, 0], new_data[i, 0], '#4EACC5', marker='.')
    # ax.set_title('Original Data')
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax = fig.add_subplot(1, 2, 2)
    # for k, col in zip(range(4), colors):
    #     my_members = k_means_labels == k
    #     # cluster_center = k_means_cluster_centers[k]
    #     ax.plot(new_data[my_members, 0], new_data[my_members, 0], 'w',markerfacecolor=col, marker='.')
    #     ax.plot(k_means_cluster_centers[k][0], k_means_cluster_centers[k][0], 'o', markerfacecolor=col,markeredgecolor='k', marker='o')
    # ax.set_title('KMeans')
    # ax.set_xticks(())
    # ax.set_yticks(())
    # plt.show()




    db = DBSCAN(eps=5).fit(new_data)
    print(db.labels_ )


    # plt.scatter(new_data,new_data)
    # plt.show()

def grey(data,label):
    data = preprocessing.MinMaxScaler().fit_transform(data)

    x_mean = np.mean(data[:,],axis=0)

    for i in range(data.shape[1]):
        data[:,i] = data[:,i] / x_mean[i]
    y = label/label.mean(axis=0)

    for i in range(data.shape[1]):
        data[:, i] = data[:,i]-y

    mmax = np.max(np.abs(data))
    mmin = np.min(np.abs(data))
    rho = 0.01

    ksi = ((mmin + rho * mmax) / (np.abs(data) + rho * mmax))
    r = np.sum(ksi,axis=0) / ksi.shape[0]

    plt.plot([i for i in range(r.shape[0])],sorted(list(r),reverse=True),label = '基于灰色关联度特征选择的重要度排序')


    # plt.bar(indices[30:], importance[30:], 1, color="yellow",label = '基于F检验特征选择的重要度')
    # plt.bar(indices[:30], importance[:30], 1, color="blue")
    #
    font1 = {
             'weight': 'normal',
             'size': 16,
             }
    #
    #
    plt.legend(prop=font1)
    plt.xlabel('特征', font1)
    plt.ylabel('重要度', font1)

    plt.show()


    return r

if __name__=='__main__':
    grey(descriptor_data,IC50)
