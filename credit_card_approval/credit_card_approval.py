#!/usr/bin/env python
# coding: utf-8

# ## Credit card applications

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1. Loading

# In[22]:


cc_apps = pd.read_csv("cc_approvals.data", header = None)
cc_apps.head()


# The probable features in a typical credit card application are Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income and finally the ApprovalStatus. 

# In[19]:


cc_apps.describe()


# In[20]:


cc_apps.info()


# Our dataset contains both numeric and non-numeric data (specifically data that are of float64, int64 and object types).
# Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.
# The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.
# Finally, the dataset has missing values, which we'll take care of in this task. The missing values in the dataset are labeled with '?', which can be seen in the last cell's output.

# ### 2. Cleaning

# In[2]:


cc_apps.tail()
cc_apps.isnull().any()


# In[3]:


import numpy as np
print(cc_apps.tail())
cc_apps = cc_apps.replace('?', np.NaN)
print(cc_apps.tail())


# In[4]:


# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)
cc_apps.isnull().sum()


# In[5]:


# Iterate over each column of cc_apps
for col in cc_apps:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

cc_apps.isnull().sum()


# In[34]:


from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

for col in cc_apps:
    if cc_apps[col].dtypes == 'object':
        cc_apps[col]=le.fit_transform(cc_apps[col])


# ### 3. Preprocessing

# In[7]:


from sklearn.model_selection import train_test_split

cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

X,y = cc_apps[:,0:13] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.33,
                                random_state=42)


# In[8]:


from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)


# ### 4. Predictive model

# In[10]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)


# ### 5. Parameter tuning

# In[13]:


from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# In[11]:


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))
confusion_matrix(y_test, y_pred)


# In[17]:


grid_model = GridSearchCV(estimator= logreg, param_grid= param_grid, cv = 5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)
best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[ ]:




