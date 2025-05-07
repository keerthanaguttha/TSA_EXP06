## Devloped by: G Keerthana
## Register Number: 212223240045
## Date: 26-04-2025
# Ex.No: 6 HOLT WINTERS METHOD
## AIM:
To implement the Holt Winters Method Model using Python

## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

## PROGRAM:
```
### Importing necessary modules

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
```
 ### Load the dataset,perform data exploration
```
data = pd.read_csv('smooth_data.csv', parse_dates=['Date'], index_col='Date')
data.head()
```
### Resample and plot data
```
data_monthly = data.resample('MS').sum()
data_monthly.plot()
```
### Scale the data and check for seasonality
```
scaler = MinMaxScaler()

scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(), index=data_monthly.index)
scaled_data.plot() # Now this plot should work correctly

decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
```
### Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate
the model predictions against test data
```
scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')

print("RMSE:", np.sqrt(mean_squared_error(test_data, test_predictions_add)))
print("Scaled Data Std Dev and Mean:", np.sqrt(scaled_data.var()), scaled_data.mean())

data_monthly = data_monthly + abs(data_monthly.min()) + 1 
```
### Create the final model and predict future data and plot it
```
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))

ax = data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Months')
ax.set_ylabel('Yahoo_price')
ax.set_title('Prediction')
plt.show()
```

## OUTPUT:
Scaled_data plot:

![image](https://github.com/user-attachments/assets/42bc62c4-0f3f-41b8-945a-2b9823a223c3)

Model performance metrics:

![image](https://github.com/user-attachments/assets/a5d8d828-706d-4624-be23-6e153d3d69cb)

Decomposed plot:

![image](https://github.com/user-attachments/assets/6a058509-062e-4afb-bb31-1d713c049a44)

Test prediction:

![image](https://github.com/user-attachments/assets/9a837c54-a23e-4302-a3b8-c2bf0b811fed)

Final prediction:

![image](https://github.com/user-attachments/assets/c6fa552e-9a81-4974-9b8c-d868f07e2316)

## RESULT:
Thus the program run successfully based on the Holt Winters Method model




