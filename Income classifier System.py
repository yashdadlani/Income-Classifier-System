import os
import pandas as pd
import numpy as np
import seaborn as sns
os.chdir("D:\Datasets")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data_income = pd.read_csv("income.csv")
data_new = data_income.copy()
print(data_new.info())
print(data_new.isnull().sum())
summary_num = data_new.describe()   #only numerical variables
print(summary_num)
summary_cate = data_new.describe(include = object)
print(summary_cate)
#Frequency of different number of categories under a variable
print(data_new['age'].value_counts())
print(data_new['JobType'].value_counts())
print(data_new['EdType'].value_counts())
print(data_new['maritalstatus'].value_counts())
print(data_new['occupation'].value_counts())
print(data_new['relationship'].value_counts())
print(data_new['capitalgain'].value_counts())
print(data_new['capitalloss'].value_counts())
print(data_new['hoursperweek'].value_counts())
print(data_new['race'].value_counts())
print(data_new['gender'].value_counts())
print(data_new['nativecountry'].value_counts())
# checking unique elements in categories of a variable
print(np.unique(data_new['age']))
print(np.unique(data_new['JobType']))
print(np.unique(data_new['EdType']))
print(np.unique(data_new['maritalstatus']))
print(np.unique(data_new['occupation']))
print(np.unique(data_new['relationship']))
print(np.unique(data_new['capitalgain']))
print(np.unique(data_new['capitalloss']))
print(np.unique(data_new['hoursperweek']))
print(np.unique(data_new['race']))
print(np.unique(data_new['gender']))
print(np.unique(data_new['nativecountry']))
data_income = pd.read_csv("income.csv",na_values=[" ?"])
data_new = data_income.copy()
print(data_new.isnull().sum())
missing = data_new[data_new.isnull().any(axis=1)]
data2 = data_new.dropna(axis=0) # this is the real data for which 
#we will work on #data2
corel = data2.corr(numeric_only =True)
#now working on relation between categorical variables
print(data2.columns)
# Gender Proportion Table
gender = pd.crosstab(index = data2['gender'],columns='count',
                      normalize=True)
gender_salstat = pd.crosstab(index= data2['gender'],
                              columns=data2['SalStat'],
                              normalize='columns')
SalStat = sns.countplot(x=data2['SalStat'],data= data2,hue = 'gender')
#Logistic Regression
data2['SalStat'] = data2['SalStat'].map({' greater than 50,000': 1,' less than or equal to 50,000': 0})
new_data = pd.get_dummies(data2,drop_first=True)
columns_list = list(new_data.columns)
features = list(set(columns_list)-set(['SalStat']))
x = new_data[features].values
y = new_data['SalStat'].values
#splitting data into train and test 
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)
logistic = LogisticRegression()
logistic.fit(train_x,train_y)
logistic.coef
logistic.intercept
