# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.



## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PANGA THANUJA
RegisterNumber:  212224040231
*/
~~~
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

~~~


## Output:
DATA HEAD:
![image](https://github.com/user-attachments/assets/936d14c1-77a0-4244-a7d8-45292abce249)

DATA INFO:
![image](https://github.com/user-attachments/assets/5c98b621-fb56-4767-9f03-278a6f5c8936)


ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/9ff56973-0222-4f71-adcc-b3d79d46ab31)


DATA HEAD FOR SALARY:
![image](https://github.com/user-attachments/assets/482d7931-5388-4544-a5e2-12dd8a754f95)

MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/fde8fca8-19f4-4311-a583-8c0493e7cce1)

R2 VALUE:

![image](https://github.com/user-attachments/assets/9d6d8ebc-ddb7-44f0-9443-a402188f95b1)


DATA PREDICTION:


![image](https://github.com/user-attachments/assets/919c8271-2d64-4241-9233-85deacd51009)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
