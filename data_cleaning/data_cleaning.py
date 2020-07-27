#!/usr/bin/env python
# coding: utf-8
Type
Null Data
Missing Data
Duplicate
Outliers
# In[19]:


get_ipython().system(' pip install pandas')


# In[2]:


import numpy as np
import pandas as pd


# ### Problem 1

# In[39]:


df = pd.read_csv('property.csv')
df.info() #Check the type of data and whether any data is null
df #Print data


# In[40]:


#Check data is null
df['ST_NUM'].isnull() 


# In[41]:


# Detecting missing values in OWN_OCCUPIED
i=0
for row in df['OWN_OCCUPIED']:
    try:
        int(row)
        df.loc[i]=np.nan
    except ValueError:
        pass
    cnt+=1
df['OWN_OCCUPIED']


# In[42]:


# Any missing values?
df.isnull().values.any()


# In[43]:


# Summarize missing values
df.isnull().sum()


# In[ ]:


# Drop all rows with missing values
df. dropna()


# In[65]:


#Replace missing values with a number 0 //fillna() fills all of null with a given value
df['ST_NUM'].fillna(0, inplace=False)
df['ST_NUM']


# In[ ]:


#Replace missing values with mean
df.replace(np.NaN, df['SQ_FT'].mean())


# In[66]:


#Replace a specific missing value with a given value
df[1,'SQ_FT'] = 1000
df['SQ_FT']


# In[69]:


#Drop duplicates
df['SQ_FT'].drop_duplicates()
df['SQ_FT']


# ### Problem 2

# In[51]:


df_nfl = pd.read_csv('nfl.csv')
df_nfl.head()


# In[52]:


df_nfl.info()


# In[56]:


#Check missing values
missing_values_count = df_nfl.isnull().sum()
missing_values_count[0:10] #See number of missing values of first 10 columns


# In[57]:


#Total missing values of all columns
total_missing = missing_values_count.sum()
total_missing


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




