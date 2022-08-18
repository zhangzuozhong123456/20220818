import numpy as np
from sklearn.model_selection import train_test_split
import xlrd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVR


X = np.transpose(np.load('data/data/GRA_deCor_DATA.npy'),(1,0))
Y = np.load('pIC50.npy')
name = np.load('data/data/GRA_deCor_NAME.npy')

# X = preprocessing.StandardScaler().fit_transform(X)
min_max_scaler = preprocessing.MinMaxScaler()
Y = min_max_scaler.fit_transform(Y.reshape(-1, 1))


def getTest(name):
    ret_x = []
    Molecular_Descriptor = r'Molecular_Descriptor.xlsx'
    Molecular_Descriptor = xlrd.open_workbook(Molecular_Descriptor)
    test_Molecular_Descriptor = Molecular_Descriptor.sheet_by_name('test')

    for id in name:
        ret_x.append(test_Molecular_Descriptor.col_values(test_Molecular_Descriptor.row_values(0).index(id))[1:])
    return np.transpose(np.array(ret_x),(1,0))


import xlwt
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('1')
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    clf = SVR(kernel='rbf')
    clf.fit(x_train, y_train)
    prds = clf.predict(x_test)
    print('MAE:%.4f MSE:%.4f RMSE:%.4f R2 Score:%.4f'%(mean_absolute_error(y_test,prds),
                                                   mean_squared_error(y_test,prds),
                                                   np.sqrt(mean_squared_error(y_test,prds)),
                                                   r2_score(y_test,prds)))

    worksheet.write(i, 0, mean_absolute_error(y_test,prds))
    # worksheet.write(j, 1, mean_squared_error(y_test,y_pred))
    worksheet.write(i, 1, np.sqrt(mean_squared_error(y_test,prds)))
    worksheet.write(i, 2, r2_score(y_test,prds))

workbook.save('data/ret/xiangxian-svr.xls')





