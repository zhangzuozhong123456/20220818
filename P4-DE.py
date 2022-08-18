import numpy as np
from sklearn.model_selection import train_test_split
import xlrd
import xlwt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sko.DE import DE
from scipy.optimize import rosen, differential_evolution


X = np.transpose(np.load('data/data/GRA_deCor_DATA.npy'),(1,0))
Y = np.load('Caco_2.npy')
CYP3A4 = np.load('CYP3A4.npy')
hERG = np.load('hERG.npy')
HOB = np.load('HOB.npy')
MN = np.load('MN.npy')

name = np.load('data/data/GRA_deCor_NAME.npy')
# X = preprocessing.StandardScaler().fit_transform(X)

def getTest(name):
    ret_x = []
    Molecular_Descriptor = r'Molecular_Descriptor.xlsx'
    Molecular_Descriptor = xlrd.open_workbook(Molecular_Descriptor)
    test_Molecular_Descriptor = Molecular_Descriptor.sheet_by_name('test')

    for id in name:
        ret_x.append(test_Molecular_Descriptor.col_values(test_Molecular_Descriptor.row_values(0).index(id))[1:])
    return np.transpose(np.array(ret_x),(1,0))



x_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), test_size=0.2)
model = xgb.XGBClassifier(use_label_encoder=False,eval_metric = 'logloss')
model.fit(x_train,y_train)

def f(x):
    x = np.array(x).reshape(1,-1)
    prds = model.predict(x)
    print(prds)
    return prds

constraint_ueq = [
    lambda x: 1 - x[0] * x[1],
    lambda x: x[0] * x[1] - 5
]

de = DE(func=f,n_dim=20)

best_x, best_y = de.run()
