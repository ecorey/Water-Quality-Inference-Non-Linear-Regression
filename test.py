import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd


# df = pd.read_csv('./data/2013_P90_Scores.csv')
# print(df.head())

file_list = ['./data/2013_P90_Scores.csv', './data/2014_P90_Scores.csv', './data/2015_P90_Scores.csv', './data/2016_P90_Scores.csv', './data/2017_P90_Scores.csv', './data/2018_P90_Scores.csv', './data/2019_P90_Scores.csv', './data/2020_P90_Scores.csv', './data/2021_P90_Scores.csv', './data/2022_P90_Scores.csv', './data/2023_P90_Scores.csv']
drop_columns = ['X', 'Y', 'x', 'y', 'GlobalID', 'OBJECTID', 'Class', 'Appd_Std', 'Restr_Std', 'Min_Date', 'Lat_DD', 'Long_DD','Grow_Area']


dataframes = []
for file in file_list:
    year = int(file.split('_')[0].split('/')[-1])  
    df = pd.read_csv(file)
    df['Year'] = year
    dataframes.append(df)


df = pd.concat(dataframes, ignore_index=True)
print(f"df shape: {df.shape}")

X = df.drop(columns=drop_columns)
y = df['P90']


# print(X.head())
# print("-------")
# print(y.head())



# encode station as it is categorical
label_encoder = LabelEncoder()
X['Station'] = label_encoder.fit_transform(X['Station'])

# print("encoded station:")
# print(df['Station'].head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Split data by year
train_years = list(range(2013, 2021))
test_year = 2021

X_train = X[df['Year'].isin(train_years)]
y_train = y[df['Year'].isin(train_years)]
X_test = X[df['Year'] == test_year]
y_test = y[df['Year'] == test_year]

# Scale After Splitting
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)





# nn
class P90InferenceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
model = P90InferenceModel(input_size)

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# training loop
epochs = 1000
for epoch in range(epochs):

    model.train()
    
    # forward pass
    y_pred = model(X_train_tensor)

    # calculate loss
    loss = criterion(y_pred, y_train_tensor) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")



# testing 
model.eval()
with torch.inference_mode():

    test_y_pred = model(X_test_tensor)
    test_loss = criterion(test_y_pred, y_test_tensor)
    print(f"\nTest Loss (MSE) for Year {test_year}: {test_loss.item():.4f}")



# vizualize visualize vizualize
y_preds = test_y_pred.numpy().flatten()
y_actual = y_test.numpy().flatten()

plt.scatter(y_actual, y_preds, alpha=0.5)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red')
plt.xlabel('Actual P90')
plt.ylabel('Predicted P90')
plt.title(f'Actual vs Predicted P90 for Year {test_year}')
plt.show()