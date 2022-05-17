import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from filter_split_scale import filter_split_scale
from sklearn.metrics import plot_confusion_matrix,classification_report,precision_score,recall_score,f1_score,accuracy_score
df=pd.read_csv('resources/final_dataset.csv')
x_train,x_test,y_train,y_test=filter_split_scale(df,'amok')
counter = Counter(y_train)
# print('Before',counter)
# print(len(x_train))
smt = SMOTE()
x_train,y_train=smt.fit_resample(x_train,y_train)
# print(len(x_train))
counter = Counter(y_train)
# print('After',counter)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train.reshape(-1))
y_pred=knn.predict(x_test)
#confusion matrix 
#matrix that is used to compare actual and predicted output values
# TP (True Positives):
# Actual positives in the data, which have been correctly predicted as positive by our model. Hence True Positive.
# TN (True Negatives):
# Actual Negatives in the data, which have been correctly predicted as negative by our model. Hence True negative.
# FP (False Positives):
# Actual Negatives in data, but our model has predicted them as Positive. Hence False Positive.
# FN (False Negatives):
# Actual Positives in data, but our model has predicted them as Negative. Hence False Negative.
from sklearn.metrics import confusion_matrix
plot_confusion_matrix(knn,x_test,y_test)
results = confusion_matrix(y_test,y_pred)
print(results)
# TP = results[0][0]
# FN = results[0][1]
# FP = results[1][0]
# TN = results[1][1]
# print('Confusion matrix')
# print(' ','Yes','No')
# print('Yes',TP,FN)
# print('No',FP,TN)

# print('###########################METRICS###########################')
# #accuracy
# accuracy = (TP+TN)/(TP+FN+FP+TN)
# print('Accuracy : ',accuracy)

# #recall or True Positive Rate
# #out of all positives how many are classified as positive
# recall = TP/(TP+FN)
# print('Recall : ',recall)

# #precision what % of classified as positive are actually positive

# precison = TP/(TP+FP)
# print('Precision : ',precison)

# #error rate: error classfication out of all classified
# err_rate = (FP+FN)/(TP+FN+FP+TN)
# print('Error rate : ',err_rate)

print("Accuracy : ",accuracy_score(y_test,y_pred))
print("Precision : ",precision_score(y_test,y_pred))
print("Recall : ",recall_score(y_test,y_pred))
print("fScore : ",f1_score(y_test,y_pred))

print("Classification report")
print(classification_report(y_test,y_pred))



