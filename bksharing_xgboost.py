## Bike sharing demand project based n boosting algos 

import pandas as pd
import numpy as np
bk=pd.read_csv("file:///E:/D Drive/R Console_class/Bike sharing data project/train.csv",parse_dates=['datetime'])
bk['year']=bk['datetime'].dt.year
bk['month']=bk['datetime'].dt.month
bk['day']=bk['datetime'].dt.day
bk['hour'] = bk['datetime'].dt.hour
bk['weekday']=bk['datetime'].dt.weekday

# converting seaoson and weather as a category as it may get treated as numbers
bk['season'] = bk['season'].astype('category')
bk['weather'] = bk['weather'].astype('category')
# dropping datetime column as we have already extracted the data and other are not needed
bk.drop(columns=['datetime','casual', 'registered'],inplace=True)

# dumming the data as 'season 'is categorical
dum_bk = pd.get_dummies(bk, drop_first=True)

X = dum_bk.drop('count',axis=1)
y = dum_bk['count']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

### XGBOOST
from xgboost import XGBRegressor
clf = XGBRegressor()
clf.fit(X_train,y_train,verbose=True)
y_pred = clf.predict(X_test)
y_pred[y_pred<0]=0
from sklearn.metrics import mean_squared_log_error
print(mean_squared_log_error(y_test,y_pred))
print[(clf.fit)]
