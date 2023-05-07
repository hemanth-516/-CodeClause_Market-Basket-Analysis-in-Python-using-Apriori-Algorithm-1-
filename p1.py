#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mlxtend


# In[3]:


#Loading neccesary packages
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[28]:


#Reading Data From Web
myretaildata = pd.read_excel(r'C:\Users\Admin\Downloads\Online_Retail.xlsx')
myretaildata.head()


# In[29]:


#Data Cleaning
myretaildata['Description'] = myretaildata['Description'].str.strip() #removes spaces from beginning and end
myretaildata.dropna(axis=0, subset=['InvoiceNo'], inplace=True) #removes duplicate invoice
myretaildata['InvoiceNo'] = myretaildata['InvoiceNo'].astype('str') #converting invoice number to be string
myretaildata = myretaildata[~myretaildata['InvoiceNo'].str.contains('C')] #remove the credit transactions 
myretaildata.head()


# In[18]:


myretaildata['Country'].value_counts()
#myretaildata.shape


# In[32]:


#Separating transactions for Germany
mybasket = (myretaildata[myretaildata['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


# In[33]:


#viewing transaction basket
mybasket.head()


# In[34]:


#converting all positive vaues to 1 and everything else to 0
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

my_basket_sets = mybasket.applymap(my_encode_units)
my_basket_sets.drop('POSTAGE', inplace=True, axis=1) #Remove "postage" as an item


# In[35]:


#Generatig frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets, min_support=0.07, use_colnames=True)


# In[36]:


#generating rules
my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)


# In[37]:


#viewing top 100 rules
my_rules.head(100)


# In[38]:


my_basket_sets['ROUND SNACK BOXES SET OF4 WOODLAND'].sum()


# In[39]:


my_basket_sets['SPACEBOY LUNCH BOX'].sum()



# In[41]:


#Filtering rules based on condition
my_rules[ (my_rules['lift'] >= 2) &
       (my_rules['confidence'] >= 0.4) ]


# In[ ]:




