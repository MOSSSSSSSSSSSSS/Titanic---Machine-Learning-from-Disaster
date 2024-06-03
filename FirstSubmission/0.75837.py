import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

titanic_data = pd.read_csv('/kaggle/input/titanic/train.csv')
# 分析数据......
import seaborn as sns
# sns.heatmap(titanic_data.corr(numeric_only=True) , cmap="YlGnBu")
# plt.show()
# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit (n_splits=1, test_size=0.2)
# for train_indices, test_indices in split. split (titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
#    strat_train_set = titanic_data.loc[train_indices]
#    strat_test_set = titanic_data.loc[test_indices] # 开发集
# plt.subplot (1,2,1)
# strat_train_set ['Survived'].hist ()
# strat_train_set['Pclass'].hist()
# plt.subplot (1,2,2)
# strat_test_set ['Survived'].hist ()
# strat_test_set ['Pclass'].hist()
# plt.show()
# 数据处理......
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X


from sklearn.preprocessing import OneHotEncoder


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        column_names = ["C", "S", "Q", "N"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        column_names = ["Female", "Male"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")


from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()),
                     ("featuredropper", FeatureDropper())])
train_set = pipeline.fit_transform(titanic_data)
train_set.info()

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = train_set.to_numpy()
        xy = xy.astype(np.float32)
        self.len = xy.shape[0]  # shape是N，9，[0]就是N
        self.x_data = torch.from_numpy(xy[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        self.y_data = torch.from_numpy(xy[:, [1]])


def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]


def __len__(self):
    return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,  # 是否打乱
                          num_workers=2)  # 几个线程去读，多线程

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(10, 6)
        self.Linear2 = torch.nn.Linear(6, 4)
        self.Linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()  # 上次是torch.nn.functional的sigmoid函数，这次是模块，可以作为一个层，这个模块没有参数，只需一个

# 也叫激活函数，这里也可以改成其他函数，改成Relu函数，relu函数取值0到1，可以是0，但在计算BCEloss是可能有对数出现，所以这里可以换成relu函数，但最后的激活函数
# 应写成sigmoid函数

def forward(self, x):
    x = self.activate(self.Linear1(x))
    x = self.activate(self.Linear2(x))
    x = self.activate(self.Linear3(x))
    return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    for index, data in enumerate(train_loader, 0):
        inputs, labels = data
        # print(inputs.shape)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


titanic_test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
final_test_data = pipeline.fit_transform(titanic_test_data)

# final_test_data.info()

X_final_test = final_test_data
X_final_test = X_final_test.fillna(method="ffill")
X_final_test = X_final_test.to_numpy()
X_final_test = X_final_test.astype(np.float32)
X_final_test
X_final_test = torch.from_numpy(X_final_test[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
y_pred = model(X_final_test)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
y_pred = y_pred.detach().numpy()
y_pred = y_pred.astype(int)
y_pred

output = pd.DataFrame(titanic_test_data['PassengerId'])
output['Survived'] = y_pred
output.to_csv("predictions.csv", index=False)
output