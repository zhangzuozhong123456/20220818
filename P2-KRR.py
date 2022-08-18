from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import xlwt

'''核岭回归'''
X = np.load('data/data/GRA_deCor_DATA.npy')
X = np.transpose(X,(1,0))
name = np.load('data/data/GRA_deCor_NAME.npy')
Y = np.load('pIC50.npy')

min_max_scaler = preprocessing.MinMaxScaler()
# X = preprocessing.StandardScaler().fit_transform(X)
Y = min_max_scaler.fit_transform(Y.reshape(-1, 1))

print('数据集维度：',X.shape)
print('标签维度：',Y.shape)


workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('1')
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    clf = KernelRidge(1)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)

    print('MAE:%.4f MSE:%.4f RMSE:%.4f R2 Score:%.4f' % (mean_absolute_error(y_test, y_hat),
                                                         mean_squared_error(y_test, y_hat),
                                                         np.sqrt(mean_squared_error(y_test, y_hat)),
                                                         r2_score(y_test, y_hat)))
    worksheet.write(i, 0, mean_absolute_error(y_test,y_hat))
    # worksheet.write(j, 1, mean_squared_error(y_test,y_pred))
    worksheet.write(i, 1, np.sqrt(mean_squared_error(y_test,y_hat)))
    worksheet.write(i, 2, r2_score(y_test,y_hat))

workbook.save('data/ret/xiangxian-krr.xls')



# r = len(x_test) + 1
# plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
# plt.plot(np.arange(1, r), y_test, 'co-', label="real")
# plt.legend()
# plt.show()