import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler


def first_attempt():

    n_in, n_h, n_out, batch_size = 10, 5, 1, 10  # Input, Hidden, Output, Batch Size
    model = nn.Sequential(
        nn.Linear(n_in, n_h),
        nn.ReLU(),
        nn.Linear(n_h, n_out),
        nn.Sigmoid()

    )
    epoch = 10
    learning_rate = 0.01
    loss = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    X = torch.randn(batch_size, n_in)
    y = torch.randn(batch_size, n_out)
    for i in range(epoch):
        y_pred = model(X)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch : {i} , Loss : {l}")


# Create A Model
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.linear(x)
        return out


def linear_regression():
    # Import data
    data = pd.read_csv("advertising.csv")
    # Take the shape of the input and the output
    # Object of scaler
    std_scalar = StandardScaler()
    x = data.iloc[ :, :-1 ].values
    x = std_scalar.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(data.iloc[ :, -1 ].values, dtype=torch.float32)
    y = y.view(-1, 1)

    input_dim = x.shape[ 1 ]
    output_dim = 1
    print(input_dim, output_dim)
    # Create a object of model

    linear_ref = LinearRegression(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(linear_ref.parameters(), lr=0.5)

    epoch = 100
    for i in range(epoch):
        y_pred = linear_ref(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(f"Epoch : {i} , Loss : {loss.item()}")

    # Train the model
    # Fit a model

   
if __name__ == "__main__":
    linear_regression()
