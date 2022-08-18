from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class DealDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]


X = np.transpose(np.load('data/DATA/GRA_deCor_DATA.npy'),(1,0))
Y = np.load('pIC50.npy')
name = np.load('data/DATA/GRA_deCor_NAME.npy')

X = preprocessing.StandardScaler().fit_transform(X)
min_max_scaler = preprocessing.MinMaxScaler()
Y = min_max_scaler.fit_transform(Y.reshape(-1, 1))


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

trainDataset = DealDataset(x_train, y_train)
train_loader = DataLoader(dataset=trainDataset, batch_size=16, shuffle=True)

testDataset = DealDataset(x_test, y_test)
test_loader = DataLoader(dataset=testDataset, shuffle=True)

model = MLP().to(device)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), 0.01)

EPOCH = 100
for epoch in range(EPOCH):
    model.train()
    loss_t = 0
    train_step = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        prds = model(inputs)
        loss = loss_func(prds, labels)
        loss.backward()
        optim.step()
        loss_t += loss.item()
        train_step += 1

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        test_step = 0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            prds = model(inputs)
            loss = loss_func(prds, labels)
            total += loss.item()
            test_step += 1
    print('Epoch %d, Train loss: %.3f Val loss: %.3f' % (epoch + 1, loss_t / train_step, total / test_step))

torch.save(model, 'model.pkl')
prd = []
with torch.no_grad():
    model.eval()
    data = torch.tensor(x_test,dtype=torch.float32).to(device)
    print(data.shape)
    prds = model(data)


prds = prds.detach().cpu().numpy()
print('MAE:%.4f MSE:%.4f RMSE:%.4f R2 Score:%.4f'%(mean_absolute_error(y_test,prds),
                                                   mean_squared_error(y_test,prds),
                                                   np.sqrt(mean_squared_error(y_test,prds)),
                                                   r2_score(y_test,prds)))


r = len(x_test) + 1
plt.plot(np.arange(1, r), prds, 'go-', label="predict")
plt.plot(np.arange(1, r), y_test, 'co-', label="real")
plt.legend()
plt.show()
