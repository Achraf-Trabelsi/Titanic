
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


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
sns.barplot(x=data_train['Pclass'],y=data_train['Survived'])
plt.show()
sns.displot(data_train['Survived'])
plt.show()

#feature engineering
data_train["Relatives"]=data_train["SibSp"]+data_train["Parch"]
data_train=data_train.drop(labels=["SibSp","Parch"],axis=1)

#train and test 
target=data_train.Survived 
data_t=data_train.drop(labels='Survived',axis=1)
x_train,x_test,y_train,y_test=train_test_split(data_t,target,test_size=0.33)
print(x_train.columns)
#preprocessing
enc=OrdinalEncoder()
x_train=enc.fit_transform(x_train)
x_test=enc.fit_transform(x_test)
#model
def  model(dtrain,ttarget,dtest,ttest,models):
    scores=[]
    acc=[]
    cross=[]
    for clf in models:
          clf.fit(dtrain,ttarget)
          pred=clf.predict(dtest)
          scores.append(clf.score(dtrain,ttarget))
          acc.append(accuracy_score(ttest,pred))
          cross.append(cross_val_score(clf,dtrain,ttarget,cv=5).mean())
          print(confusion_matrix(ttest,pred))

    res=pd.DataFrame({'Model':models,'Score':scores,'Accuracy':acc,'Cross score':cross})
    return res
        
model_1=KNeighborsClassifier(n_neighbors=10,weights="distance")
model_2=MultinomialNB()
model_3=RandomForestClassifier(n_estimators=5)
models=[model_1,model_2,model_3]
print(model(x_train,y_train,x_test,y_test,models))


    
    

          

    

    
 