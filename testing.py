
from asyncore import read
from importlib.resources import path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from model import x_train,y_train
data_test=pd.read_csv("test.csv")
target=pd.read_csv("gender_submission.csv")
passID=target["PassengerId"]
print(passID)
target=target.drop(labels=['PassengerId'],axis=1)
target=np.ravel(target)
print(data_test.shape)
data_test=data_test.drop(labels=['PassengerId','Cabin','Name','Ticket'],axis=1)
data_test['Age'].fillna(data_test['Age'].mean(),inplace=True)
data_test['Fare'].fillna(data_test['Fare'].mean(),inplace=True)
missing_values=data_test.isnull().sum()
print(missing_values)
data_test["Relatives"]=data_test["SibSp"]+data_test["Parch"]
data_test=data_test.drop(labels=["SibSp","Parch"],axis=1)
#preprocessing
enc=OrdinalEncoder()
data_test=enc.fit_transform(data_test)
model_3=RandomForestClassifier(n_estimators=5)

model_3.fit(x_train,y_train)
pred=model_3.predict(data_test)
print("confusion matrix: ", confusion_matrix(target,pred))
print("accuracy score: " , accuracy_score(target,pred))

sub=pd.DataFrame({"PassengerID":passID,"Survived":pred})
print(sub.shape)
sub.to_csv(path_or_buf="submission.csv",index=False)
x=pd.read_csv("submission.csv")
print(x.shape)