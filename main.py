import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#21 tane sector
#61 tane industry
#262 tane hqcity

df = pd.read_csv("fortune_global_500_from_2019-2021.csv")


print(df.info())
df.drop(["permalink","hqstate","name","hqcity","industry"],axis=1,inplace=True)
print(df.columns)
      
prf = float((df["prftchange"].sum()) / len(df["prftchange"]))
prof = float((df["profits"].sum()) / len(df["profits"]))
rev = float((df["revchange"].sum()) / len(df["revchange"]))

df["prftchange"].fillna(prf,inplace=True)
df["profits"].fillna(prof,inplace=True)
df["revchange"].fillna(rev,inplace=True)


#newcomer
#profitable
#ceowoman
#jobgrowth
#no=0 / yes=1

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df["newcomer"] = le.fit_transform(df["newcomer"])
df["profitable"] = le.fit_transform(df["profitable"])
df["ceowoman"] = le.fit_transform(df["ceowoman"])
df["jobgrowth"] = le.fit_transform(df["jobgrowth"])
df["sector"] = le.fit_transform(df["sector"]) 

X = df.drop(["assets"],axis=1)
Y= df["assets"]

"""y = df.iloc[:,7].values
df.drop(["assets"],axis=1,inplace=True)
x = df.iloc[:,:12].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
"""

logreg = LogisticRegression()
logreg.fit(X, Y)
y_pred = logreg.predict(X)



