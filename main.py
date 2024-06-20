import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
concrete_data.describe
concrete_data.isnull().sum()
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] 
target = concrete_data['Strength'] 
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
n_cols = predictors.shape[1] 
def regressionModel():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=n_cols))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
model = regressionModel()
MSE = []
STD = []
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    model.fit(x_test, y_train, epochs=100)
    yhat = model.predict(x_test)
    mse = mean_squared_error(y_test, yhat)
    MSE.append(mse)
    std = mse - yhat
    STD.append(std)
for mSE in MSE:
    print(mSE)
for sTD in STD:
    print(sTD)