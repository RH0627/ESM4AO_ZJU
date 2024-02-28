#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from numpy import loadtxt
import statistics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import math


def evaluation(clf_in,X,y):
    AUC_collection = []
    BACC_collection = []
    Recall_collection = []
    Precision_collection = []
    MCC_collection = []
    F1_collection = []
    for i in range(10):
    # dataset splitting
        X_train_whole, X_ind_test, y_train_whole, y_ind_test =  train_test_split(X, y, test_size=0.2, random_state=i)
    # clf = LogisticRegression(penalty= 'none', C=i, max_iter = 2000)
        clf = clf_in
        clf.fit(X_train_whole,y_train_whole)# fitting model 
        y_pred = clf.predict(X_ind_test)    # predict results
        y_true = y_ind_test                 # asign values for confusiong matrix calculation
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
        if isinstance(clf,LinearSVC):
            AUC = roc_auc_score(y_true,clf.decision_function(X_ind_test))
        elif isinstance(clf,SVC):
            AUC = roc_auc_score(y_true,clf.decision_function(X_ind_test))
        else:
            AUC = roc_auc_score(y_true,clf.predict_proba(X_ind_test)[:,1])
    

    
        AUC_collection.append(AUC)
        Recall = TP/(TP+FN)
        Recall_collection.append(Recall)
        Precision = TP/(TP+FP)
        Precision_collection.append(Precision)
        MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
        MCC_collection.append(MCC)
        F1 = (2*Recall*Precision)/(Precision+Recall)
        F1_collection.append(F1)
        BACC_collection.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
    
    
    
    print("BACC = ",round(statistics.mean(BACC_collection),3),'±',round(statistics.stdev(BACC_collection),3))
    print("Recall = ",round(statistics.mean(Recall_collection),3),'±',round(statistics.stdev(Recall_collection),3))
    print("Precision = ",round(statistics.mean(Precision_collection),3),'±',round(statistics.stdev(Precision_collection),3))
    print("MCC = ",round(statistics.mean(MCC_collection),3),'±',round(statistics.stdev(MCC_collection),3))
    print("F1 score = ",round(statistics.mean(F1_collection),3),'±',round(statistics.stdev(F1_collection),3)) 
    print("ROC_AUC = ",round(statistics.mean(AUC_collection),3),'±',round(statistics.stdev(AUC_collection),3)) 
            
    return clf           
            


# In[ ]:




