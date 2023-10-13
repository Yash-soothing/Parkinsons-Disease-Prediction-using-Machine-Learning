
# Project Title : Parkinsons Disease Prediction using Machine Learning

Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages to it and affects more than 1 million individuals every year in India. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain. The dataset has 24 columns and 195 records and is only 39.7 KB, we can get the dataset through kaggle.




## Steps to implement in the project are :-

-  we need to import te dependencies

- Data Collection and Analysis

- We need to label the data 

- Data Pre-Processing

- Model Training



## Usage/Examples
Importing the Dependencies

```javascript
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```

Data Collection & Analysis
```javascript
parkinsons_data = pd.read_csv('/content/parkinsons.csv')
parkinsons_data.head()
parkinsons_data.shape
parkinsons_data.info()
parkinsons_data.isnull().sum()
parkinsons_data.describe()
parkinsons_data['status'].value_counts()
parkinsons_data.groupby('status').mean()
```


Data Pre-Processing
```javascript
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
```

Data Standardization
```javascript
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
print(X_train)
```

Model Training
```javascript
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
```

Model Evaluation
```javascript
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of test data : ', test_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data : ', test_data_accuracy)
```

Building a Predictive System
```javascript
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")

```


## Screenshots

![App Screenshot](https://camo.githubusercontent.com/62fc96b357c894ed69cee28de86968412c6d0865214e1ab72865f3bf2976dfd9/68747470733a2f2f6d656469612e6973746f636b70686f746f2e636f6d2f766563746f72732f7061726b696e736f6e732d646973656173652d73796d70746f6d732d766563746f722d6964383831373032343236)

