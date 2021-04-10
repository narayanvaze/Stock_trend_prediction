#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

HDFC_Working = pd.read_csv(r'C:\Users\MOHIT.K\Desktop\HDFC_Train.csv')

HDFC_Working["Open"] = minmax_scale(HDFC_Working["Open"], (0,1))
HDFC_Working["High"] = minmax_scale(HDFC_Working["High"], (0,1))
HDFC_Working["Low"] = minmax_scale(HDFC_Working["Low"], (0,1))
HDFC_Working["Close"] = minmax_scale(HDFC_Working["Close"], (0,1))
HDFC_Working["Volume"] = minmax_scale(HDFC_Working["Volume"], (0,1))
HDFC_Working["Candle Body"] = minmax_scale(HDFC_Working["Candle Body"], (0,1))
HDFC_Working["Upper Shadow"] = minmax_scale(HDFC_Working["Upper Shadow"], (0,1))
HDFC_Working["Lower Shadow"] = minmax_scale(HDFC_Working["Lower Shadow"], (0,1))
HDFC_Working["Range"] = minmax_scale(HDFC_Working["Range"], (0,1))
HDFC_Working["RSI_7"] = minmax_scale(HDFC_Working["RSI_7"], (0,1))
HDFC_Working["RSI_14"] = minmax_scale(HDFC_Working["RSI_14"], (0,1))
HDFC_Working["RSI_21"] = minmax_scale(HDFC_Working["RSI_21"], (0,1))
HDFC_Working["MACD (26, 12)"] = minmax_scale(HDFC_Working["MACD (26, 12)"], (0,1))
HDFC_Working["MACD (44, 26)"] = minmax_scale(HDFC_Working["MACD (44, 26)"], (0,1))
HDFC_Working["OBV"] = minmax_scale(HDFC_Working["OBV"], (0,1))
HDFC_Working["ATR_7"] = minmax_scale(HDFC_Working["ATR_7"], (0,1))
HDFC_Working["ATR_14"] = minmax_scale(HDFC_Working["ATR_14"], (0,1))
HDFC_Working["ATR_21"] = minmax_scale(HDFC_Working["ATR_21"], (0,1))
HDFC_Working["AO"] = minmax_scale(HDFC_Working["AO"], (0,1))
HDFC_Working["KAMA"] = minmax_scale(HDFC_Working["KAMA"], (0,1))
HDFC_Working["PPO"] = minmax_scale(HDFC_Working["PPO"], (0,1))
HDFC_Working["PPO_Signal"] = minmax_scale(HDFC_Working["PPO_Signal"], (0,1))
HDFC_Working["PVO"] = minmax_scale(HDFC_Working["PVO"], (0,1))
HDFC_Working["PVO_Signal"] = minmax_scale(HDFC_Working["PVO_Signal"], (0,1))
HDFC_Working["ROC"] = minmax_scale(HDFC_Working["ROC"], (0,1))
HDFC_Working["TSI"] = minmax_scale(HDFC_Working["TSI"], (0,1))
HDFC_Working["WR"] = minmax_scale(HDFC_Working["WR"], (0,1))
HDFC_Working["ADI"] = minmax_scale(HDFC_Working["ADI"], (0,1))
HDFC_Working["CMI"] = minmax_scale(HDFC_Working["CMI"], (0,1))
HDFC_Working["FI"] = minmax_scale(HDFC_Working["FI"], (0,1))
HDFC_Working["MFI"] = minmax_scale(HDFC_Working["MFI"], (0,1))
HDFC_Working["NVI"] = minmax_scale(HDFC_Working["NVI"], (0,1))
HDFC_Working["VPT"] = minmax_scale(HDFC_Working["VPT"], (0,1))
HDFC_Working["ADX"] = minmax_scale(HDFC_Working["ADX"], (0,1))
HDFC_Working["MASS INDEX"] = minmax_scale(HDFC_Working["MASS INDEX"], (0,1))

train_features =  HDFC_Working.drop(['Target', 'Candle Score'], axis = 1)
train_labels = HDFC_Working["Target"].copy()
train_labels_1 = (train_labels == 1)
train_labels_0 = (train_labels == 0)


HDFC_Working = pd.read_csv(r'C:\Users\MOHIT.K\Desktop\HDFC_Valid.csv')

HDFC_Working["Open"] = minmax_scale(HDFC_Working["Open"], (0,1))
HDFC_Working["High"] = minmax_scale(HDFC_Working["High"], (0,1))
HDFC_Working["Low"] = minmax_scale(HDFC_Working["Low"], (0,1))
HDFC_Working["Close"] = minmax_scale(HDFC_Working["Close"], (0,1))
HDFC_Working["Volume"] = minmax_scale(HDFC_Working["Volume"], (0,1))
HDFC_Working["Candle Body"] = minmax_scale(HDFC_Working["Candle Body"], (0,1))
HDFC_Working["Upper Shadow"] = minmax_scale(HDFC_Working["Upper Shadow"], (0,1))
HDFC_Working["Lower Shadow"] = minmax_scale(HDFC_Working["Lower Shadow"], (0,1))
HDFC_Working["Range"] = minmax_scale(HDFC_Working["Range"], (0,1))
HDFC_Working["RSI_7"] = minmax_scale(HDFC_Working["RSI_7"], (0,1))
HDFC_Working["RSI_14"] = minmax_scale(HDFC_Working["RSI_14"], (0,1))
HDFC_Working["RSI_21"] = minmax_scale(HDFC_Working["RSI_21"], (0,1))
HDFC_Working["MACD (26, 12)"] = minmax_scale(HDFC_Working["MACD (26, 12)"], (0,1))
HDFC_Working["MACD (44, 26)"] = minmax_scale(HDFC_Working["MACD (44, 26)"], (0,1))
HDFC_Working["OBV"] = minmax_scale(HDFC_Working["OBV"], (0,1))
HDFC_Working["ATR_7"] = minmax_scale(HDFC_Working["ATR_7"], (0,1))
HDFC_Working["ATR_14"] = minmax_scale(HDFC_Working["ATR_14"], (0,1))
HDFC_Working["ATR_21"] = minmax_scale(HDFC_Working["ATR_21"], (0,1))
HDFC_Working["AO"] = minmax_scale(HDFC_Working["AO"], (0,1))
HDFC_Working["KAMA"] = minmax_scale(HDFC_Working["KAMA"], (0,1))
HDFC_Working["PPO"] = minmax_scale(HDFC_Working["PPO"], (0,1))
HDFC_Working["PPO_Signal"] = minmax_scale(HDFC_Working["PPO_Signal"], (0,1))
HDFC_Working["PVO"] = minmax_scale(HDFC_Working["PVO"], (0,1))
HDFC_Working["PVO_Signal"] = minmax_scale(HDFC_Working["PVO_Signal"], (0,1))
HDFC_Working["ROC"] = minmax_scale(HDFC_Working["ROC"], (0,1))
HDFC_Working["TSI"] = minmax_scale(HDFC_Working["TSI"], (0,1))
HDFC_Working["WR"] = minmax_scale(HDFC_Working["WR"], (0,1))
HDFC_Working["ADI"] = minmax_scale(HDFC_Working["ADI"], (0,1))
HDFC_Working["CMI"] = minmax_scale(HDFC_Working["CMI"], (0,1))
HDFC_Working["FI"] = minmax_scale(HDFC_Working["FI"], (0,1))
HDFC_Working["MFI"] = minmax_scale(HDFC_Working["MFI"], (0,1))
HDFC_Working["NVI"] = minmax_scale(HDFC_Working["NVI"], (0,1))
HDFC_Working["VPT"] = minmax_scale(HDFC_Working["VPT"], (0,1))
HDFC_Working["ADX"] = minmax_scale(HDFC_Working["ADX"], (0,1))
HDFC_Working["MASS INDEX"] = minmax_scale(HDFC_Working["MASS INDEX"], (0,1))

valid_features =  HDFC_Working.drop(['Target', 'Candle Score'], axis = 1)
valid_labels = HDFC_Working["Target"].copy()
valid_labels_1 = (valid_labels == 1)
valid_labels_0 = (valid_labels == 0)


HDFC_Working = pd.read_csv(r'C:\Users\MOHIT.K\Desktop\HDFC_Test.csv')

HDFC_Working["Open"] = minmax_scale(HDFC_Working["Open"], (0,1))
HDFC_Working["High"] = minmax_scale(HDFC_Working["High"], (0,1))
HDFC_Working["Low"] = minmax_scale(HDFC_Working["Low"], (0,1))
HDFC_Working["Close"] = minmax_scale(HDFC_Working["Close"], (0,1))
HDFC_Working["Volume"] = minmax_scale(HDFC_Working["Volume"], (0,1))
HDFC_Working["Candle Body"] = minmax_scale(HDFC_Working["Candle Body"], (0,1))
HDFC_Working["Upper Shadow"] = minmax_scale(HDFC_Working["Upper Shadow"], (0,1))
HDFC_Working["Lower Shadow"] = minmax_scale(HDFC_Working["Lower Shadow"], (0,1))
HDFC_Working["Range"] = minmax_scale(HDFC_Working["Range"], (0,1))
HDFC_Working["RSI_7"] = minmax_scale(HDFC_Working["RSI_7"], (0,1))
HDFC_Working["RSI_14"] = minmax_scale(HDFC_Working["RSI_14"], (0,1))
HDFC_Working["RSI_21"] = minmax_scale(HDFC_Working["RSI_21"], (0,1))
HDFC_Working["MACD (26, 12)"] = minmax_scale(HDFC_Working["MACD (26, 12)"], (0,1))
HDFC_Working["MACD (44, 26)"] = minmax_scale(HDFC_Working["MACD (44, 26)"], (0,1))
HDFC_Working["OBV"] = minmax_scale(HDFC_Working["OBV"], (0,1))
HDFC_Working["ATR_7"] = minmax_scale(HDFC_Working["ATR_7"], (0,1))
HDFC_Working["ATR_14"] = minmax_scale(HDFC_Working["ATR_14"], (0,1))
HDFC_Working["ATR_21"] = minmax_scale(HDFC_Working["ATR_21"], (0,1))
HDFC_Working["AO"] = minmax_scale(HDFC_Working["AO"], (0,1))
HDFC_Working["KAMA"] = minmax_scale(HDFC_Working["KAMA"], (0,1))
HDFC_Working["PPO"] = minmax_scale(HDFC_Working["PPO"], (0,1))
HDFC_Working["PPO_Signal"] = minmax_scale(HDFC_Working["PPO_Signal"], (0,1))
HDFC_Working["PVO"] = minmax_scale(HDFC_Working["PVO"], (0,1))
HDFC_Working["PVO_Signal"] = minmax_scale(HDFC_Working["PVO_Signal"], (0,1))
HDFC_Working["ROC"] = minmax_scale(HDFC_Working["ROC"], (0,1))
HDFC_Working["TSI"] = minmax_scale(HDFC_Working["TSI"], (0,1))
HDFC_Working["WR"] = minmax_scale(HDFC_Working["WR"], (0,1))
HDFC_Working["ADI"] = minmax_scale(HDFC_Working["ADI"], (0,1))
HDFC_Working["CMI"] = minmax_scale(HDFC_Working["CMI"], (0,1))
HDFC_Working["FI"] = minmax_scale(HDFC_Working["FI"], (0,1))
HDFC_Working["MFI"] = minmax_scale(HDFC_Working["MFI"], (0,1))
HDFC_Working["NVI"] = minmax_scale(HDFC_Working["NVI"], (0,1))
HDFC_Working["VPT"] = minmax_scale(HDFC_Working["VPT"], (0,1))
HDFC_Working["ADX"] = minmax_scale(HDFC_Working["ADX"], (0,1))
HDFC_Working["MASS INDEX"] = minmax_scale(HDFC_Working["MASS INDEX"], (0,1))

test_features =  HDFC_Working.drop(['Target', 'Candle Score'], axis = 1)
test_labels = HDFC_Working["Target"].copy()
test_labels[544] = 1
test_labels_1 = (test_labels == 1)
test_labels_0 = (test_labels == 0)

cross_val_test_features = valid_features.copy()

cross_val_test_labels = valid_labels.copy()


# In[22]:


### POLY_5 SVM C = 0.001 ####

filename = 'POLY_5_SVM_CLF_1.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[20]:


### POLY_4 SVM C = 10 ####

filename = 'POLY_4_SVM_CLF_4.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[4]:


### POLY_2 SVM C = 0.1 ####

filename = 'POLY_2_SVM_CLF_2.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[17]:


### SIGMOID SVM C = 10 ####

filename = 'SIG_SVM_CLF_4.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[21]:


### RBF SVM C = 1 ####

filename = 'RBF_SVM_CLF_3.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[25]:


### LIN SVM C = 100 ####

filename = 'LIN_SVM_CLF_5.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[24]:


### RBF SVM C = 1.25 ####

filename = 'RBF_SVM_CLF_6.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[ ]:





# In[16]:


### POLY_3 SVM C = 10 ####

filename = 'POLY_3_SVM_CLF_4.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[3]:


### RANDOM FOREST ####

filename = 'RF.sav'
loaded_model = pickle.load(open(filename, 'rb'))
preds = loaded_model.predict(test_features)

cm = confusion_matrix(test_labels, preds)
print(cm)
print("\n")
print("Precision =")
print(precision_score(test_labels, preds ))
print("\n")
print("Recall =")
print(recall_score(test_labels, preds))
print("\n")
print("F1 =")
print(f1_score(test_labels, preds))
print("\n")
print("Accuracy =")
print(accuracy_score(test_labels, preds))
print("\n")
print("Negative Precision =") 
Neg_Prec = cm[0,0]/(cm[0,0] + cm[1,0])
print(Neg_Prec)
print("\n")
print("Specificity =")
spec = cm[0,0]/(cm[0,0] + cm[0,1])
print(spec)
print("\n")
print("F2 =")
f2 = 2*(Neg_Prec*spec)/(Neg_Prec + spec)
print(f2)


# In[ ]:





# In[ ]:




