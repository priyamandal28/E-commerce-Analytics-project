#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataset = pd.read_csv("C:\\Users\\home\\Downloads\\E-com_Data.csv")
dataset.head()


# In[3]:


dataset.info()


# In[4]:


dataset.nunique()


# In[5]:


print(dataset.isnull().sum()/len(dataset)*100)


# In[6]:


dataset.duplicated().sum()


# In[7]:


dataset= dataset.rename(columns = {'InvoieNo':'InvoiceNo','Date of purchase':'Date'})


# In[8]:


dataset.head()


# In[9]:


dataset.isnull().sum()


# In[10]:


dataset = dataset.drop_duplicates(ignore_index = True)


# In[11]:


dataset.isnull().sum()


# In[12]:


#dropping the missing customer ID


# In[13]:


dataset = dataset.dropna(subset = ['CustomerID'])


# In[14]:


dataset.isnull().sum()


# In[15]:


dataset.head()


# In[16]:


#changung the date format to yyyy-mm-dd


# In[17]:


dataset['Date'] = pd.to_datetime(dataset['Date'])


# In[18]:


dataset.head()


# In[19]:


#creating one more date column of date for EDA purpose


# In[20]:


dataset['Date1']=dataset['Date']


# In[21]:


dataset.head()


# In[22]:


dataset.info()


# In[23]:


dataset['Date'].describe()


# In[24]:


#here min shows the first date and max shows the last date


# In[25]:


# Recency = Latest date - Last invoice date
# Frequency = count of invoice no of transaction(s)
# Monetary = Sum of Total

import datetime as dt

latest_date = dt.datetime(2017,12,20)


# In[26]:


RFMScores = dataset.groupby('CustomerID').agg({'Date1':lambda x:(latest_date-x.max()).days,
                                              'Date': lambda x:x.count(),
                                              'Price': lambda x:x.sum()})

#cConverting the date into int for understanding purpose
RFMScores['Date']=RFMScores['Date'].astype(int)

#rename columns as Recency,frequency,Monetory

RFMScores.rename(columns={'Date1':'Recency','Date':'Frequency','Price':'Monetory'},inplace = True)

RFMScores.reset_index().head()


# In[27]:


RFMScores.columns


# In[28]:


RFMScores['Recency'].describe()


# In[29]:


#splitting the data into four segments using quantile method
quantiles = RFMScores.quantile(q=[0.25,0.50,0.75])
quantiles = quantiles.to_dict()


# In[30]:


quantiles


# In[31]:


#creating own function to define R,F and M segment

def RScoring(x,p,d):
    if x<= d[p][0.25]:
        return 1
    elif x<= d[p][0.50]:
        return 2
    elif x<= d[p][0.75]:
        return 3
    else:
        return 4
def FnMScoring(x,p,d):
    if x<=d[p][0.25]:
        return 4
    elif x<= d[p][0.50]:
        return 3
    elif x<= d[p][0.75]:
        return 2
    else:
        return 1


# In[32]:


RFMScores.columns


# In[33]:


RFMScores['R']=RFMScores['Recency'].apply(RScoring,args =('Recency',quantiles,))
RFMScores['F']=RFMScores['Frequency'].apply(FnMScoring,args =('Frequency',quantiles,))
RFMScores['M']=RFMScores['Monetory'].apply(FnMScoring,args =('Monetory',quantiles,))


# In[34]:


RFMScores.head(20)


# In[35]:


RFMScores['RFMGroup']=RFMScores.R.map(str)+RFMScores.F.map(str)+RFMScores.M.map(str)


# In[37]:


RFMScores.head()


# In[38]:


RFMScores['RFMScore']=RFMScores[['R','F','M']].sum(axis = 1)


# In[39]:


RFMScores.head(10)


# In[43]:


#Assigning loyalty to each customer

Loyalty_level = ['Diamond','Platinum','Gold','Silver']
score_cuts = pd.qcut(RFMScores.RFMScore,q = 4,labels = Loyalty_level)
RFMScores['RFMScore_Loyalty_level']=score_cuts.values
RFMScores.reset_index().head(20)


# In[44]:


#validating the data for RFMGroup = 111

RFMScores[RFMScores['RFMGroup']=='111'].sort_values('Monetory',ascending = False).reset_index().head()


# In[46]:


RFMScores=RFMScores.reset_index()


# In[ ]:


#handling the negatuve values present in data

def handling_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
#applying handling_neg_n_zero function to recency and monetory columns
RFMScores['Recency']=[handling_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetory']=[handling_neg_n_zero(x) for x in RFMScores.Monetory]


# In[47]:


#Perform Log transformation to bring data into normal or near normal distribution
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetory']].apply(np.log, axis = 1).round(3)


# In[48]:


Log_Tfd_Data.head()


# In[49]:


new_data = RFMScores[['Recency','Frequency','Monetory']]


# In[50]:


new_data.head()


# # Feature Scaling

# In[51]:


sns.boxplot(y = 'Monetory', data=RFMScores)


# In[ ]:


#there are outliers so we are performing standardization technique


# In[52]:


from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
scaled_data = sca.fit_transform(new_data)
scaled_data


# In[53]:


scaled_data = pd.DataFrame(scaled_data,index = RFMScores.index,columns = new_data.columns)


# In[54]:


scaled_data.head()


# # Build clustering model

# In[55]:


from sklearn.cluster import KMeans

sum_of_sq_dist = {}

for k in range(1,15):
    km = KMeans(n_clusters = k, init='k-means++', max_iter=300)
    km=km.fit(scaled_data)
    sum_of_sq_dist[k] = km.inertia_
    
# Visualisation for getting elbow method (to find the actual k value)

sns.pointplot(x = list(sum_of_sq_dist.keys()),
             y = list(sum_of_sq_dist.values()))
plt.xlabel("Number of Clusters(k)")
plt.ylabel("Sum of Square Distance(Euclidean Distance)")
plt.title("Elbow Method for Optimal K value")
plt.show()


# In[56]:


km = KMeans(n_clusters = 4, init='k-means++', max_iter=300)


# In[57]:


y_kmeans = km.fit_predict(scaled_data)


# In[58]:


y_kmeans


# In[59]:


RFMScores['Cluster'] = km.labels_
RFMScores.head()


# In[60]:


RFMScores.info()


# In[61]:


RFMScores.to_excel('Final_output.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




