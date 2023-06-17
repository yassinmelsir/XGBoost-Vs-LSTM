import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable 
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def performancefigure(predictions, actual):
    predicted_emissions = predictions[:,2]
    actual_emissions = actual[:,2]

    plt.figure(figsize=(10,6))
    plt.plot(predicted_emissions,label='Predicted')
    plt.plot(actual_emissions,label='Actual')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions per Year')
    plt.legend()
    plt.show()

def predictionsfigure(predictions, actual):
    predicted_year = predictions[:,1]
    predicted_emissions = predictions[:,2]

    actual_year = actual[:,1]
    actual_emissions = actual[:,2]

    plt.figure(figsize=(10,6))
    plt.plot(predicted_year,predicted_emissions,label='Predicted')
    plt.plot(actual_year,actual_emissions,label='Actual')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.title('Actual vs Predicted Emissions per Year')
    plt.legend()
    plt.show()

def encode(df,X,y):
    def encode_country(country):
        return label_encoder.transform([country])[0]

    def decode_country(country):
        return label_encoder.inverse_transform([country])[0]
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Country'])
    X['Country'] = X['Country'].apply(encode_country)
    y['Country'] = y['Country'].apply(encode_country)

    return X, y

def model(input_size):
    input_size = 29 #number of features
    hidden_size = 2 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers

    num_classes = 3 #number of output classes
    return LSTM1(num_classes, input_size, hidden_size, num_layers, input_size) #our lstm class
 
def lstm_init(X,y):
  mm = MinMaxScaler()
  ss = StandardScaler()

  ss.fit(X)
  mm.fit(y)

  X_ss = ss.fit_transform(X)
  y_mm = mm.fit_transform(y) 


  X_train, X_test, y_train, y_test = train_test_split(X_ss, y_mm, random_state=1)

  X_train_tensors = Variable(torch.Tensor(X_train))
  X_test_tensors = Variable(torch.Tensor(X_test))

  y_train_tensors = Variable(torch.Tensor(y_train))
  y_test_tensors = Variable(torch.Tensor(y_test))     

  X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
  X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

  print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
  print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 
  
  lstm1 = model(X_train_tensors_final.shape[1])

  num_epochs = 1000 #1000 epochs
  learning_rate = 0.001 #0.001 lr


  criterion = torch.nn.MSELoss()    # mean-squared error for regression
  optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

  for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0
  
    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)
  
    loss.backward() #calculates the loss of the loss function
  
    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

    torch.save(lstm1.state_dict(), '/Users/yme/code/AppliedAI/summativeassessment/lstm/model.pth')
    
def lstm_predict(X,y):
  mm = MinMaxScaler()
  ss = StandardScaler()

  ss.fit(X)
  mm.fit(y)

  df_X_ss = ss.transform(X) #old transformers
  df_y_mm = mm.transform(y) #old transformers

  df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
  df_y_mm = Variable(torch.Tensor(df_y_mm))

  #reshaping the dataset
  df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 

  lstm1 = model(df_X_ss.shape[1])

  lstm1.load_state_dict(torch.load('/Users/yme/code/AppliedAI/summativeassessment/lstm/model.pth'))

  train_predict = lstm1(df_X_ss)#forward pass
  data_predict = train_predict.data.numpy() #numpy conversion
  dataY_plot = df_y_mm.data.numpy()

  data_predict = mm.inverse_transform(data_predict) #reverse transformation
  dataY_plot = mm.inverse_transform(dataY_plot)

  return data_predict, dataY_plot

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out