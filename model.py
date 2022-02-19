import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OrdinalEncoder
#understanding data
data_train=pd.read_csv("train.csv")
#print(data_train.describe().to_string())
print(data_train.shape)
print(data_train.columns)
#sns.displot(data_train['Survived'])
plt.show()
#imputing missing values
data_train=data_train.drop(labels=['PassengerId','Cabin','Name','Ticket'],axis=1)
data_train['Age'].fillna(data_train['Age'].mean(),inplace=True)
data_train['Embarked'].fillna(data_train['Embarked'].mode()[0],inplace=True)
missing_values=data_train.isnull().sum()
print(missing_values)
#visualize
corr=data_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

#train and test 
target=data_train.Survived 
data_t=data_train.drop(labels='Survived',axis=1)
x_train,x_test,y_train,y_test=train_test_split(data_t,target,test_size=0.33)
#preprocessing
enc=OrdinalEncoder()
x_train=enc.fit_transform(x_train)
x_test=enc.fit_transform(x_test)
#model
