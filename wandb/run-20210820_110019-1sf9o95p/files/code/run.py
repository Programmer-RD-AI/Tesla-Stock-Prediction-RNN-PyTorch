import torch
import numpy as np
import pandas as pd
from torch.nn import *
from torch.optim import *
from help_funcs import *
from model import *
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    LabelEncoder,
    Normalizer,
)

data = pd.read_csv("./data/data.csv")
data = data["High"]
data.dropna(inplace=True)
data = torch.from_numpy(np.array(data.tolist()))
data_input = data.view(1, -1)[:1, :-1].to(device).float()
data_target = data.view(1, -1)[:1, 1:].to(device).float()
model = Model()
criterion = MSELoss()
optimizer = LBFGS(model.parameters(), lr=0.8)
name = "baseline"
epochs = 50
# model = train(
#     optimizer, criterion, model, data_input, data_target, name=name, epochs=epochs
# )
preprocessings = [
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    OneHotEncoder,
    LabelEncoder,
    Normalizer,
]
for preprocessing in preprocessings:
    model = Model().to(device)
    criterion = MSELoss()
    optimizer = LBFGS(model.parameters(), lr=0.8)
    name = f'{preprocessing()}-preprocessing'
    data = pd.read_csv("./data/data.csv")
    data = data["High"]
    data.dropna(inplace=True)
    preprocessing = preprocessing()
    preprocessing.fit(np.array(data).reshape(-1,1))
    data = preprocessing.transform(np.array(data).reshape(-1,1))
    data = np.array(data.tolist())
    data = torch.from_numpy(data)
    data_input = data.view(1, -1)[:1, :-1].to(device).float()
    data_target = data.view(1, -1)[:1, 1:].to(device).float()
    model = train(
        optimizer, criterion, model, data_input, data_target, name=name, epochs=epochs
    )
