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


print(df.info)

print(df.columns)
df["year"]=df["year"].astype(int)
df["rank"]=df["rank"].astype(int)
df["sector"]=df["sector"].astype(int)
df["revenues"]=df["revenues"].astype(int)
df["revchange"]=df["revchange"].astype(int)
df["profits"]=df["profits"].astype(int)
df["prftchange"]=df["prftchange"].astype(int)

df["assests"]=df["assets"].astype(int)
df["employees"]=df["employees"].astype(int)
df["newcomer"]=df["newcomer"].astype(int)
df["profitable"]=df["profitable"].astype(int)
df["ceowoman"]=df["ceowoman"].astype(int)
df["jobgrowth"]=df["jobgrowth"].astype(int)

X = df.drop(["assets"],axis=1).values
Y= df["assets"].values

print(df.columns)

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
print("-------------------------")

print(r_dt.predict([[2019,1,4,514405,27,3,5845,81000,1,1,1,1,1]]))
