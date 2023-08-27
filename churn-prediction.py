# Importing Libraries

# Utitlity Libraries
import numpy as np
import pandas as pd

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Algorithm, Evaluation, and Model Libraries 
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Loading Data

# Importing Dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Printing Data
data.head()
data.shape
data.isna().sum
data.info()
data.describe()
with sns.color_palette("pastel"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    sns.countplot(x="gender", data=data, ax=axes[0,0])
    sns.countplot(x="SeniorCitizen", data=data, ax=axes[0,1])
    sns.countplot(x="Partner", data=data, ax=axes[0,2])
    sns.countplot(x="Dependents", data=data, ax=axes[1,0])
    sns.countplot(x="PhoneService", data=data, ax=axes[1,1])
    sns.countplot(x="PaperlessBilling", data=data, ax=axes[1,2])
with sns.color_palette("pastel"):
    sns.countplot(x="InternetService", data=data)
with sns.color_palette("pastel"):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    sns.countplot(x="StreamingTV", data=data, ax=axes[0,0])
    sns.countplot(x="StreamingMovies", data=data, ax=axes[0,1])
    sns.countplot(x="OnlineSecurity", data=data, ax=axes[0,2])
    sns.countplot(x="OnlineBackup", data=data, ax=axes[1,0])
    sns.countplot(x="DeviceProtection", data=data, ax=axes[1,1])
    sns.countplot(x="TechSupport", data=data, ax=axes[1,2])
with sns.color_palette("pastel"):
    plt.figure(figsize=(10,6))
    sns.countplot(x="Contract", data=data)
with sns.color_palette("pastel"):
    plt.figure(figsize=(10,6))
    sns.countplot(x="PaymentMethod", data=data)
with sns.color_palette("husl"):
    fig, axes = plt.subplots(1,2, figsize=(12, 7))
    sns.histplot(data["tenure"], ax=axes[0],kde=True)
    sns.histplot(data["MonthlyCharges"], ax=axes[1],kde=True)
churnnum = {'Yes':1, 'No':0}
data.Churn.replace(churnnum, inplace=True)
churnnum
genderval = pd.pivot_table(data, values='Churn', index=['gender'],
                    columns=['SeniorCitizen'], aggfunc=np.mean)
genderval
data.drop(['customerID','gender','Contract','TotalCharges'], axis=1, inplace=True)
data.columns

minmax = MinMaxScaler()
a = minmax.fit_transform(data[['tenure']])
b = minmax.fit_transform(data[['MonthlyCharges']])
data['tenure'] = a
data['MonthlyCharges'] = b

data.shape
# Resampling
nox = data[data.Churn == 0]
noy = data[data.Churn == 1]

yesupsampled = noy.sample(n=len(nox), replace=True, random_state=42)
print(len(yesupsampled))
data.PhoneService.value_counts()

data.MultipleLines.value_counts()
from sklearn.model_selection import train_test_split
X = yesupsampled.drop(['Churn','PaymentMethod'], axis=1) #features (independent variables)
y = yesupsampled['Churn'] #target (dependent variable)
columns_to_convert = ['PhoneService', 'Partner', 'Dependents','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','InternetService']
X[columns_to_convert] = (X[columns_to_convert] == 'Yes').astype(int)
print("length of data-",len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=50)
X_train

# Ridge Classifier

# Ceeating a model
ridgeclassifier = RidgeClassifier() 

# Fitting Classifier
ridgeclassifier.fit(X_train, y_train) 

# Predicting training values
pred = ridgeclassifier.predict(X_train)

# Getting accuracy
accuracy_score(y_train, pred)

# Predicting testing values
pred_test = ridgeclassifier.predict(X_test)

# Printing accuracy
accuracy_score(y_test, pred_test)
# Creating Model
randomforest = RandomForestClassifier(n_estimators=250, max_depth=18)

# Fitting Model
randomforest.fit(X_train, y_train)
# Pridicting training accuracy
pred = randomforest.predict(X_train)

# Printing training accuracy
accuracy_score(y_train, pred)

# Predicting testing accuracy
pred_test = randomforest.predict(X_test)

# Printing testing score
accuracy_score(y_test, pred_test)
# Grid Search

# Setting parameters
parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}

# Using Grid Search with Random Forest Classifier
forest = RandomForestClassifier()
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)

# Fitting the model
clf.fit(X, y)
# Getting the best parameters and score
clf.best_params_
{'max_depth': 20, 'n_estimators': 150}
clf.best_score_
# Grid Search To Get Best Hyperparameters
parameters = {'C':[0.01,0.1,1,3,5,10]}
svmclf = SVC(class_weight='balanced',random_state=43)
grid = GridSearchCV(estimator=svmclf, param_grid=parameters,scoring='accuracy',return_train_score=True,verbose=1)
grid.fit(X_train,y_train)

# Plotting the values
cv_result = pd.DataFrame(grid.cv_results_)
plt.scatter(cv_result['param_C'],cv_result['mean_train_score'])
plt.plot(cv_result['param_C'],cv_result['mean_train_score'],label='Train')
plt.scatter(cv_result['param_C'],cv_result['mean_test_score'])
plt.plot(cv_result['param_C'],cv_result['mean_test_score'],label="CV")
plt.title('Hyperparameter vs accuracy')
plt.legend()
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()
# Training the model using the optimal parameters discovered with SVM Classifier
svmclf =  SVC(C=3,class_weight='balanced', random_state=43)
svmclf.fit(X_train,y_train)

result2 = ["2.","SVM","Balanced using class weights"]
y_pred_tr = svmclf.predict(X_train)
print('Train accuracy SVM: ',accuracy_score(y_train,y_pred_tr))
result2.append(round(accuracy_score(y_train,y_pred_tr),2))

y_pred_test = svmclf.predict(X_test)
print('Test accuracy SVM: ',accuracy_score(y_test,y_pred_test))
result2.append(round(accuracy_score(y_test,y_pred_test),2))

recall = recall_score(y_test,y_pred_test)
print("Recall Score: ",recall)
result2.append(round(recall,2))

# Building a confusion matrix
matrix = confusion_matrix(y_test,y_pred_test)
ax=plt.subplot();
sns.heatmap(matrix, annot=True, fmt='d', linewidths=2, linecolor='black', cmap='YlGnBu',ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_ylim(2.0,0)
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Neg','Pos'])
ax.yaxis.set_ticklabels(['Neg','Pos'])
plt.show()

import math
scale=round(math.sqrt(y_train.value_counts()[0]/y_train.value_counts()[1]),2)

# Grid Search To Get Best Hyperparameters
parameters = {"learning_rate"    : [0.10,0.20,0.30 ],\
              "max_depth"        : [ 3,5,10,20],\
              "n_estimators" : [ 100, 200, 300, 500],\
              "colsample_bytree" : [ 0.3, 0.5, 0.7 ] }
clf_xgb = XGBClassifier(scale_pos_weight=scale, eval_metric ='mlogloss')
grid = GridSearchCV(estimator=clf_xgb, param_grid=parameters, scoring='accuracy',return_train_score=True,verbose=1)
grid.fit(X_train,y_train)

# plotting only the first 70 train scores
cv_result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_train_score',ascending=True)[:70]
param_list = list(cv_result['params'])
param_index = np.arange(70)
plt.figure(figsize=(18,6))
plt.scatter(param_index,cv_result['mean_train_score'])
plt.plot(param_index,cv_result['mean_train_score'],label='Train')
plt.scatter(param_index,cv_result['mean_test_score'])
plt.plot(param_index,cv_result['mean_test_score'],label="CV")
plt.title('Hyperparameter vs accuracy')
plt.grid()
plt.legend()
plt.xlabel('Hyperparametr combination Dict')
plt.ylabel('Accuracy')
plt.show()

best_parameters = param_list[34]
print(best_parameters)

# Using XG Boost
clf_xgb = XGBClassifier(learning_rate= best_parameters['learning_rate'] ,max_depth=best_parameters ['max_depth'], n_estimators=best_parameters['n_estimators'], colsample_bytree=best_parameters['colsample_bytree'],                        eval_metric='mlogloss',scale_pos_weight=scale)
clf_xgb.fit(X_train,y_train)

xgbresult = ["4.","XGBClassifier","Balanced using scale_pos_weight"]
y_pred_tr = clf_xgb.predict(X_train)
print('Train accuracy XGB: ',accuracy_score(y_train,y_pred_tr))
xgbresult.append(round(accuracy_score(y_train,y_pred_tr),2))

y_pred_test = clf_xgb.predict(X_train)
print('Test accuracy XGB: ',accuracy_score(y_test,y_pred_test))
xgbresult.append(round(accuracy_score(y_test,y_pred_test),2))

recall = recall_score(y_test,y_pred_test)
print("Recall Score: ",recall)
xgbresult.append(round(recall,2))

# Building confusion matrix
cm = confusion_matrix(y_test,y_pred_test)
ax=plt.subplot();
sns.heatmap(cm, annot=True, fmt='d', linewidths=2, linecolor='black', cmap='YlGnBu',ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_ylim(2.0,0)
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Neg','Pos'])
ax.yaxis.set_ticklabels(['Neg','Pos'])
plt.show()

