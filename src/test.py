from pyexpat.model import XML_CQUANT_REP
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from prometheus_client import Counter
from filter_split_scale import filter_split_scale
df=pd.read_csv('resources/final_dataset.csv')
x_train,x_test,y_train,y_test=filter_split_scale(df,'amok')
counter = Counter(x_train)
print('Before',counter)
smt = SMOTE()
x_train,y_train=smt.fit_resample(x_train,y_train)
counter = Counter(x_train)
print('After',counter)


