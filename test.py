import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def prepare_data(df, forecast_col, forecast_out, test_size):
    # creating new column called label with the last 5 rows are NaN
    label = df[forecast_col].shift(-forecast_out)
    # creating the feature array
    X = np.array(df[[forecast_col]])
    # processing the feature array
    X = preprocessing.scale(X)
    # creating the column i want to use later in the predicting method
    X_lately = X[-forecast_out:]
    # X that will contain the training and testing
    X = X[:-forecast_out]
    # dropping na values
    label.dropna(inplace=True)
    # assigning y
    y = np.array(label)
    # cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    response = [X_train, X_test, y_train, y_test, X_lately]
    return response


df = pd.read_csv('Data/prices.csv')
forecast_col = 'Close'
forecast_out = 5
test_size = 0.2

# calling the method were the cross validation and data preparation is in
X_train, X_test, y_train, y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)
# initializing linear regression model
learner = LinearRegression()
# training the linear regression model
learner.fit(X_train, y_train)

# testing the linear regression model
score = learner.score(X_test, y_test)
# set that will contain the forecasted data
forecast0 = learner.predict(X_test)
forecast = learner.predict(X_lately)

# for i in range(len(X_test)):
#     print(forecast0[i], y_test[i])

# creating json object
response = {'test_score': score, 'forecast_set': forecast}

print(response)
