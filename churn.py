#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import OrdinalEncoder



# In[2]:


#uploading the telco customer churn  data
df=pd.read_csv("telco.csv (2).zip")


# In[3]:


#viewing the first rows of data
df.head()


# In[4]:


df


# #Data exploration

# In[5]:


df.describe()#displaying the summary statistics of the data



# In[6]:


df.info() # to print information about the data


# In[7]:


df.columns


# In[8]:


df.describe(include=["bool","object"])


# In[9]:


#checking  null values
df.notnull().sum()


# In[10]:


df.Population


# In[11]:


df.shape


# #Explolatory Data analysis

# In[12]:


#vizualizing histogram of age distribution
sns.histplot(df['Age'],bins=10,kde=True)
plt.title("Age  distribution")
plt.show()

           


# In[13]:


#checking the  count of a specific Age 
age_count=df[df['Age']>=80].shape[0]
age_count


# In[14]:


#grouping ange into Age group
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# In[15]:


df.Age


# In[16]:


df


# In[17]:


#A graph for count of age groups
sns.countplot(x='Age',data=df,color='pink')
plt.title("count of age groups")
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[18]:


churn_count=df[df['Churn Score']>=6].shape[0]
churn_count


# In[19]:


non_numeric_churn_scores = df[~df['Churn Score'].apply(lambda x: isinstance(x, (int, float)))]
print(non_numeric_churn_scores)


# In[20]:


bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
df['Churn Score'] = pd.cut(df['Churn Score'], bins=bins, labels=labels, right=False)


# In[21]:


#A graph of churn scores 
sns.countplot(x='Churn Score',data=df,)
plt.title("count ofChurn Score")
plt.xlabel("Churn Score")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[30]:


sns.boxplot(x='Churn Label', y='Satisfaction Score', data=df,color='purple')
plt.title('Satisfaction Score vs. Churn')
plt.show()


# In[23]:


#churn of  married against  customer status
plt.figure(figsize=(10, 6))
sns.countplot(x='Customer Status', hue='Married', data=df, palette='muted')
plt.title('Churn by Marriage status')
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.legend(title='Married', loc='upper right')
plt.xticks(rotation=45)
plt.show()


# In[24]:


#churn graph by churn reasons
plt.figure(figsize=(10, 6))
sns.countplot(x='Customer Status', hue='Churn Reason', data=df, palette='dark')
plt.title('Churn by Churn Reason')
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.legend(title='Churn Reason', loc='upper right')
plt.xticks(rotation=45)
plt.show()


# In[25]:


#churn graph by churn category
plt.figure(figsize=(10, 6))
sns.countplot(x='Customer Status', hue='Churn Category', data=df, palette='dark')
plt.title('Churn by Churn Category')
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.legend(title='Churn Category', loc='upper right')
plt.xticks(rotation=45)
plt.show()


# In[26]:


sns.countplot(x='Contract', hue='Churn Label', data=df, palette='viridis')
plt.title('Contract Type vs. Churn')
plt.show()


# In[27]:


plt.figure(figsize=(10,6))
sns.countplot(x='Customer Status',hue='Age',data=df,palette='muted')
plt.title('customer churn by Age group')
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.legend(title='Age group',loc='upper right')
plt.xticks(rotation=45)
plt.show()

           


# In[28]:


columns_df = pd.DataFrame({
    'Index': range(len(df.columns)),
    'Column Name': df.columns
})

print(columns_df)


# In[35]:


# View the column by index number
column_index =30 # Index of the  column
column_data = df.iloc[:, column_index]
print(column_data)


# In[ ]:




