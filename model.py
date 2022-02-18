import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

#understanding data
data_train=pd.read_csv("train.csv")
print(data_train.describe().to_string())
print(data_train.shape)
print(data_train.columns)
missing_values=data_train.isnull().sum()
print(missing_values)

sns.displot(data_train['Survived'])
plt.show()
