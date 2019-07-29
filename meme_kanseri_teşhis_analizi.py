# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 02:08:10 2019

@author: kevser
"""


import numpy as np # linear algebra
import pandas as pd # data processing xlsx file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph

from sklearn.linear_model import LogisticRegression # to apply the Logistic Regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import cross_val_score # use for cross validation
from sklearn.model_selection import StratifiedKFold # use for cross validation in past version
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model

from sklearn.metrics import confusion_matrix 
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs


data=pd.read_csv("dataset.csv",sep=",")
print(data.head(2))
data.info()#sahip olduğumuz veri türüne bakıyoruz burda işe yaramayan verileri görüp silebiliriz
data.drop("Unnamed: 32",axis=1,inplace=True) #işe yaramayan datadan kurtuluyoruz
data.columns
print(data.columns.tolist())
data.drop("id",axis=1,inplace=True)#id ye gerek yok ondan kurtulduk
print(data.head (2))#kontrol etmek için bastırdım
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.describe()
sns.countplot(data['diagnosis'],label="Count")
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)


train_X = train[prediction_var] #eğitim için veri girişi
train_y=train.diagnosis #eğitim için çıktılar

test_X= test[prediction_var] #test için veri girişi
test_y =test.diagnosis #test için çıktılar


model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y) #veri izleme modelimize uyuyor random forest classsifier 

prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

#SVM de deneyelim

model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)



prediction_var = features_mean #Tüm özellikler için yapalım
train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

#tüm özelllikleri kullandığımızda özelliklerin eğitime etkisini görmek için degerleri sıraladık.
#Random forest kullandığımızda nasıl oldugunu gördük
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)

#şimdi tüm özellikleri svm kullandıgımızda nasıl bi dogruluk degeri alacagımızı görelim.
model = svm.SVC( gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
#SVM nin dogrulugunda azalma çok fazla şimdi RFC de en iyi sonuc veren 5 özelliği kullanıcaz 
prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean'] 
  
#eğitim ve test datalarını alıyoruz.
train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

#RFC uygulanıyor

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


#SVM uygulanıyor

model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

#en kötü özellikle verilere bakalım
prediction_var = features_worst

train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

 #svm çalısıyor
model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

#rfc çalısıyor
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
   
#özelliklerin önem sırasını gördük
featim = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
print(featimp)

#en önemli 5 özelliği alıyoruz 

prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 

train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

#check for SVM
model = svm.SVC(gamma='auto')
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)


color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15, 15));


features_mean
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
def model(model,data,prediction,outcome):
    
    kf = KFold(data.shape[0], n_folds=10)
def classification_model(model,data,prediction_input,output):
    model.fit(data[prediction_input],data[output]) #Here we fit the model using training set
  
    #Make predictions on training set:
    predictions = model.predict(data[prediction_input])
  
    #Print accuracy
    # now checkin accuracy for same data
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    kf = KFold(data.shape[0], n_folds=5)
    # About cross validitaion please follow this link
    #https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
    #let me explain a little bit data.shape[0] means number of rows in data
    #n_folds is for number of folds
    error = []
    for train, test in kf:
        # as the data is divided into train and test using KFold
        # now as explained above we have fit many models 
        # so here also we are going to fit model
        #in the cross validation the data in train and test will change for evry iteration
        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data
        # here iloc[train,:] means all row in train in kf amd the all columns
        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train
        # Training the algorithm using the predictors and target.
        model.fit(train_X, train_y)
    
        # now do this for test data also
        test_X=data[prediction_input].iloc[test,:]
        test_y=data[output].iloc[test]
        error.append(model.score(test_X,test_y))
        # printing the score 
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
        model = DecisionTreeClassifier()
        prediction_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
        outcome_var= "diagnosis"
        classification_model(model,data,prediction_var,outcome_var)
        model = svm.SVC()

        classification_model(model,data,prediction_var,outcome_var)
        
        model = KNeighborsClassifier()
        classification_model(model,data,prediction_var,outcome_var)
        
        model = RandomForestClassifier(n_estimators=100)
        classification_model(model,data,prediction_var,outcome_var)
        
        model=LogisticRegression()
        classification_model(model,data,prediction_var,outcome_var)
data_X= data[prediction_var]
data_y= data["diagnosis"]
        
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):
    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")
    clf.fit(train_X,train_y)
    print("The best parameter found on development set is :")
            # this will gie us our best parameter to use
    print(clf.best_params_)
    print("the bset estimator is ")
    print(clf.best_estimator_)
    print("The best score is ")
            # this is the best score that we can achieve using these parameters#
    print(clf.best_score_)
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': [2,3,4,5,6,7,8,9,10], 
              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }
    model= DecisionTreeClassifier()
    Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
            
model = KNeighborsClassifier()

k_range = list(range(1, 23))
leaf_size = list(range(1,23))
weight_options = ['uniform', 'distance']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)

model=svm.SVC()
param_grid = [
        {'C': [1, 10, 100, 1000], 
         'kernel': ['linear']
         },
        {'C': [1, 10, 100, 1000], 
         'gamma': [0.001, 0.0001], 
         'kernel': ['rbf']
         },
        ]
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        
        
        
        
























































