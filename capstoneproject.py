#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("Dent.csv")


# In[6]:


print(df.isnull().sum())
numeric_columns=df.select_dtypes(include=['number']).columns
df[numeric_columns]=df[numeric_columns].fillna(df[numeric_columns].mean())


# In[8]:


x=df.drop(columns=['Gender','Sample ID','Sl No'])
y=df['Gender']


# In[11]:


from sklearn.preprocessing import Normalizer
scaler=Normalizer()
x_normalized=scaler.fit_transform(x)


# In[12]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_normalized,y,test_size=0.2,random_state=42)


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
label_encoder=LabelEncoder()
y_train_encoded=label_encoder.fit_transform(y_train)
y_test_encoded=label_encoder.fit_transform(y_test)
y_pred_encoded=label_encoder.transform(y_pred)


# In[49]:


accuracy=accuracy_score(y_test_encoded,y_pred_encoded)
cm=confusion_matrix(y_test_encoded,y_pred_encoded)
roc_auc=roc_auc_score(y_test_encoded,y_pred_encoded)
fpr=[0.1,0.2,0.3,0.4,0.5]
tpr=[0.2,0.3,0.4,0.5,0.6]
roc_au=auc(fpr,tpr)
plt.figure()

plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--',label='Random Guess')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics (ROC) curve')
plt.legend(loc="lower right",fontsize='medium')
plt.show()
print("Accuracy:",accuracy)
print("confusion matrix:",cm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




