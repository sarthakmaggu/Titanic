# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:03:59 2019

@author: Sarthak Maggu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#Handlng Missing Values and loading dataset
data = pd.read_csv("train.csv")
#null = data.isna().sum()
data.drop("PassengerId", axis =1, inplace = True) #droping PassengerId
data["Sex"] = LabelEncoder().fit_transform(data["Sex"]) # Label Encoding for Categorical Variables
data["Age"] = data["Age"].fillna(data["Age"].mean()) # Filling Nan with mean
data["Name"] = data["Name"].map(lambda x:  x.split(',')[1].split('.')[0].strip()) #Filtering out titles
titles = data['Name'].unique()
data.drop("Ticket", axis =1 ,inplace = True) #droping Tickets
#count = data["Embarked"].value_counts(), calculating frequency of the embarked group.
data["Embarked"] = data["Embarked"].fillna("S")
#counts = data["Cabin"].value_counts()
data["Cabin"] = data["Cabin"].fillna("C")
data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
#counts1 = data["Cabin"].value_counts()
replacement = {
    'Don': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 6,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}
data['Name'] = data['Name'].apply(lambda x: replacement.get(x))
data['Name'] = StandardScaler().fit_transform(data['Name'].values.reshape(-1,1))
data['Pclass'] = StandardScaler().fit_transform(data['Pclass'].values.reshape(-1,1))
data['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1,1))
data['SibSp'] = StandardScaler().fit_transform(data['SibSp'].values.reshape(-1,1))
data['Parch'] = StandardScaler().fit_transform(data['Parch'].values.reshape(-1,1))
data['Fare'] = StandardScaler().fit_transform(data['Fare'].values.reshape(-1,1))
replacement1 = {
    'S': 0,
    'Q': 1,
    'C': 2
}
data['Embarked'] = data['Embarked'].apply(lambda x: replacement1.get(x))
data['Embarked'] = StandardScaler().fit_transform(data['Embarked'].values.reshape(-1,1))
replacement2 = {
      'C': 0,
      'B': 1,
      'D': 2,
      'E': 3,
      'A': 4,
      'F': 5,
      'G': 6,
      'T': 7
        }
data['Cabin'] = data['Cabin'].apply(lambda x: replacement2.get(x))
data['Cabin'] = StandardScaler().fit_transform(data['Cabin'].values.reshape(-1,1))

data1 = pd.read_csv("test.csv")
data1.drop("PassengerId", axis =1, inplace = True) #droping PassengerId
data1["Sex"] = LabelEncoder().fit_transform(data1["Sex"]) # Label Encoding for Categorical Variables
data1["Age"] = data1["Age"].fillna(data1["Age"].mean()) # Filling Nan with mean
data1["Name"] = data1["Name"].map(lambda x:  x.split(',')[1].split('.')[0].strip()) #Filtering out titles
data1["Name"] = data1["Name"].fillna("Mr")
#titles = data1['Name'].value_counts()
data1.drop("Ticket", axis =1 ,inplace = True) #droping Tickets
#count = data["Embarked"].value_counts(), calculating frequency of the embarked group.
data1["Embarked"] = data1["Embarked"].fillna("S")
#counts = data["Cabin"].value_counts()
data1["Cabin"] = data1["Cabin"].fillna("C")
data1['Cabin'] = data1['Cabin'].apply(lambda x: x[0])
#counts1 = data["Cabin"].value_counts()
data1['Fare'] = data1['Fare'].fillna(data1["Fare"].mean())
c = data1.isna().sum()
replacement = {
    'Don': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 6,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}
data1['Name'] = data1['Name'].apply(lambda x: replacement.get(x))
data1['Name'] = StandardScaler().fit_transform(data1['Name'].values.reshape(-1,1))
data1['Pclass'] = StandardScaler().fit_transform(data1['Pclass'].values.reshape(-1,1))
data1['Age'] = StandardScaler().fit_transform(data1['Age'].values.reshape(-1,1))
data1['SibSp'] = StandardScaler().fit_transform(data1['SibSp'].values.reshape(-1,1))
data1['Parch'] = StandardScaler().fit_transform(data1['Parch'].values.reshape(-1,1))
data1['Fare'] = StandardScaler().fit_transform(data1['Fare'].values.reshape(-1,1))
replacement1 = {
    'S': 0,
    'Q': 1,
    'C': 2
}
data1['Embarked'] = data1['Embarked'].apply(lambda x: replacement1.get(x))
data1['Embarked'] = StandardScaler().fit_transform(data1['Embarked'].values.reshape(-1,1))
replacement2 = {
      'C': 0,
      'B': 1,
      'D': 2,
      'E': 3,
      'A': 4,
      'F': 5,
      'G': 6,
      'T': 7
        }
data1['Cabin'] = data1['Cabin'].apply(lambda x: replacement2.get(x))
data1['Cabin'] = StandardScaler().fit_transform(data1['Cabin'].values.reshape(-1,1))
data1['Name'] = data1['Name'].fillna(data1['Name'].mean())
X = data.iloc[:, 0:9]
y = data['Survived']
y = y[:, np.newaxis]

model = RandomForestRegressor(n_estimators = 100)
model.fit(X, y.ravel())
ypred = model.predict(data1)



