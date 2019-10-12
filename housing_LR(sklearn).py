import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Reading Dataset
data=pd.read_csv('housing.csv')
df=pd.DataFrame(data)
print(df.head())

#Eliminating Null values
print(df.shape)
df.dropna(axis=0, inplace=True)
print(df.shape)

#reshaping
X=df.iloc[:,4].values.reshape(-1,1)
Y=df.iloc[:,5].values.reshape(-1,1)

#Predicting
lr=LinearRegression()
lr.fit(X,Y)
Y_predict=lr.predict(X)

#Ploting
plt.scatter(X,Y)
plt.plot(X, Y_predict, color="red")
plt.show()