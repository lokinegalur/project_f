import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def filter_split_scale(df,type):
    df_mod1 = df[df['class'] == 'normal'] #or 'class' == typ]
    df_mod2 = df[df['class'] == type]
    df = df_mod1.append(df_mod2)
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,:-1], test_size=0.20, random_state=42,shuffle=True)
    sc = MinMaxScaler()
    x_train  = pd.DataFrame(sc.fit_transform(x_train), columns =['frame.time_delta_displayed','wlan.fc.type','wlan.duration'])
    x_test = pd.DataFrame(sc.transform(x_test), columns =['frame.time_delta_displayed','wlan.fc.type','wlan.duration',])
    y_train = pd.DataFrame(y_train,columns=['class'])
    y_test = pd.DataFrame(y_test,columns=['class'])
    return x_train,x_test,y_train,y_test
