import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
res = Path('./', 'resources')
data = pd.read_csv('resources/preproc_dataset.csv')
drop_cols = ['radiotap.flags.shortgi','radiotap.rxflags.badplcp','wlan.fc.frag','wlan.fc.moredata','wlan.fcs_good']
data = data.drop(drop_cols,axis=1)

data.to_csv(
    Path(res, 'final_dataset.csv'),
    index=False,
    sep=',')


'''x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.33,random_state=1)
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = pd.DataFrame(x_train, columns =['frame.time_delta_displayed','wlan.fc.type','wlan.duration'])
x_test = pd.DataFrame(x_test, columns =['frame.time_delta_displayed','wlan.fc.type','wlan.duration',])
y_train = pd.DataFrame(y_train,columns=['class'])
y_test = pd.DataFrame(y_test,columns=['class'])

res_final = Path('./', 'resources/final_datasets')

x_train.to_csv(
    Path(res_final, 'x_train.csv'),
    index=False,
    sep=',')

x_test.to_csv(
    Path(res_final, 'x_test.csv'),
    index=False,
    sep=',')

y_train.to_csv(
    Path(res_final, 'y_train.csv'),
    index=False,
    sep=',')

y_test.to_csv(
    Path(res_final, 'y_test.csv'),
    index=False,
    sep=',') '''

