import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


veriler =pd.read_csv('veriler.csv')

data = veriler.iloc[:,1:4].values
cinsiyet = veriler.iloc[:,4:].values

data_train, data_test, cinsiyet_train, cinsiyet_test = train_test_split(data,cinsiyet,test_size=0.33,random_state=0)

svc = SVC(kernel='linear', gamma=1)
#gamma can make 2d plane 3d in order to make data more separetable
svc.fit(data_train,cinsiyet_train)

y_pred = svc.predict(data_test)

cm =confusion_matrix(cinsiyet_test,y_pred)

print(cm)

