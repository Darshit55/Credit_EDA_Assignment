#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Reading csv Files

# In[2]:


appli_train = pd.read_csv('application_data.csv')
prev_appli = pd.read_csv('previous_application.csv')


# In[3]:


appli_train.shape


# In[4]:


appli_train.columns


# In[5]:


appli_train.describe()


# In[6]:


appli_train.columns.values


# In[7]:


appli_train.count()


# In[8]:


len(appli_train)


# In[9]:


appli_train.isna().sum()


# ## Null Values in a column

# In[10]:


appli_train.isna().sum().sort_values(ascending=False).head(60)


# In[11]:


appli_train.shape


# Only keep those rows which have less than 45% data missing. 100/45 = 2.22

# In[12]:


a = len(appli_train)/2.22
a


# In[13]:


type(appli_train)


# In[14]:


#only keep those rows which have less than 45% data missing
appli_train.columns[appli_train.isna().sum()<138518]


# In[15]:


(138518/len(appli_train))*100


# In[16]:


#Only keep those rows which have less than 45% data missing
appli_train = appli_train[appli_train.columns[appli_train.isna().sum()<138518]]
appli_train


# In[17]:


appli_train.head()


# The occupation type have maximum number of nulls below 45% i.e 96391

# In[18]:


appli_train.isna().sum().sort_values(ascending=False).head(20)


# In[19]:


appli_train.select_dtypes(include='object').columns


# In[20]:


appli_train.dtypes


# In[21]:


appli_train['OCCUPATION_TYPE'].head()


# In[22]:


appli_train['OCCUPATION_TYPE'].mode()


# In[23]:


appli_train['OCCUPATION_TYPE'].unique()


# In[24]:


appli_train['OCCUPATION_TYPE'].value_counts()


# After marking countplot it can be seen that maximum number of customers are with occupation labourers.

# In[25]:


label= ['Laborers', 'Core staff', 'Accountants', 'Managers',
       'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff',
       'Private service staff', 'Medicine staff', 'Security staff',
       'High skill tech staff', 'Waiters/barmen staff',
       'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff',
       'HR staff']
sns.countplot(x = 'OCCUPATION_TYPE', data = appli_train).set_xticklabels(labels=label,rotation=90)
plt.show()


# In[26]:


appli_train['EXT_SOURCE_3'].head(20)


# In[27]:


appli_train['EXT_SOURCE_3'].mean()


# In[28]:


appli_train['EXT_SOURCE_3'].median()


# It can be seen from the histogram that in column EXT_SOURCE_3 maximum number of values are between 0.5 to 0.75.

# In[29]:


plt.hist(appli_train['EXT_SOURCE_3'], color='skyblue', edgecolor='black')
plt.show()


# In[30]:


appli_train['EXT_SOURCE_3'].fillna(appli_train['EXT_SOURCE_3'].median(),inplace=True)


# In[31]:


appli_train['EXT_SOURCE_3'].isna().sum()


# In[32]:


plt.hist(appli_train['EXT_SOURCE_3'], color='skyblue', edgecolor='black')
plt.show()


# In[33]:


appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].head()


# It can be seen that maximum number of values in column 'AMT_REQ_CREDIT_BUREAU_YEAR' are between 0.0 and 3.0.

# In[34]:


plt.hist(appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'],color='brown', edgecolor='white')
plt.show()


# In[35]:


appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts()


# In[36]:


appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()


# In[37]:


appli_train['AMT_REQ_CREDIT_BUREAU_YEAR']= appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].mode().iloc[0])


# In[38]:


appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'].isna().sum()


# In[39]:


plt.hist(appli_train['AMT_REQ_CREDIT_BUREAU_YEAR'],color='brown', edgecolor='white')
plt.show()


# In[40]:


appli_train['AMT_REQ_CREDIT_BUREAU_WEEK'].head()


# In[41]:


appli_train['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts()


# In[42]:


plt.hist(appli_train['AMT_REQ_CREDIT_BUREAU_WEEK'],color='red', edgecolor='white')
plt.show()


# As Column has only almost single value and its 0.0, so it cant give us varible insights. Its better to drop this column

# In[43]:


del(appli_train['AMT_REQ_CREDIT_BUREAU_WEEK'])


# ## Similarly treating other columns

# In[44]:


appli_train['NAME_TYPE_SUITE'].value_counts()


# In[45]:


appli_train['NAME_TYPE_SUITE'].isna().sum()


# In[46]:


appli_train['NAME_TYPE_SUITE'].unique()


# In[47]:


label= ['Unaccompanied', 'Family', 'Spouse, partner', 'Children',
       'Other_A', 'Other_B', 'Group of people']
sns.countplot(x = 'NAME_TYPE_SUITE', data = appli_train).set_xticklabels(labels=label,rotation=90)
plt.show()


# In[48]:


list(set(appli_train.columns)-set(appli_train.describe().columns))


# In[49]:


cols_00=list(set(appli_train.columns)-set(appli_train.describe().columns))
appli_train[cols_00]=appli_train[cols_00].fillna(appli_train.mode().iloc[0])


# In[50]:


appli_train[cols_00].isna().sum()


# In[51]:


nulls=appli_train.isnull().sum()
nulls[nulls>0]


# In[52]:


appli_train.isna().sum().sort_values(ascending=False).head(15)


# In[53]:


appli_train['AMT_REQ_CREDIT_BUREAU_QRT'].head()


# In[54]:


appli_train['AMT_REQ_CREDIT_BUREAU_QRT'].head()


# In[55]:


sns.displot(appli_train['AMT_REQ_CREDIT_BUREAU_QRT'])
plt.show()


# As Column has only almost single value and its 0.0, so it cant give us varible insights. Its better to drop this column

# In[56]:


del(appli_train['AMT_REQ_CREDIT_BUREAU_QRT'])


# In[57]:


appli_train[['AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR']].head()


# In[58]:


appli_train[['AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR']].describe()


# In[59]:


sns.displot(appli_train['AMT_REQ_CREDIT_BUREAU_MON'])
plt.show()


# In[60]:


sns.displot(appli_train['AMT_REQ_CREDIT_BUREAU_DAY'])
plt.show()


# In[61]:


sns.displot(appli_train['AMT_REQ_CREDIT_BUREAU_HOUR'])
plt.show()


# As All Three Columns have only almost single value and its 0.0, so it cant give us varible insights. Its better to drop these columns

# In[62]:


appli_train['AMT_REQ_CREDIT_BUREAU_MON']=appli_train['AMT_REQ_CREDIT_BUREAU_MON'].fillna(appli_train['AMT_REQ_CREDIT_BUREAU_MON'].median())
appli_train['AMT_REQ_CREDIT_BUREAU_DAY']=appli_train['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(appli_train['AMT_REQ_CREDIT_BUREAU_DAY'].median())
appli_train['AMT_REQ_CREDIT_BUREAU_HOUR']=appli_train['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(appli_train['AMT_REQ_CREDIT_BUREAU_HOUR'].median())


# In[63]:


appli_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']].describe()


# As All Four Columns have only almost single value and its 0.0, so it cant give us varible insights. Its better to drop these columns

# In[64]:


appli_train.drop(columns=['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE'],inplace=True)


# In[65]:


appli_train['AMT_ANNUITY'].head()


# In[66]:


sns.displot(appli_train['AMT_ANNUITY'])
plt.show()


# In[67]:


appli_train['AMT_ANNUITY'].describe()


# In[68]:


appli_train['AMT_ANNUITY']=appli_train['AMT_ANNUITY'].fillna(appli_train['AMT_ANNUITY'].median())


# In[69]:


appli_train['AMT_ANNUITY'].isna().sum()


# In[70]:


b=pd.DataFrame(appli_train.isna().sum().sort_values(ascending=False).head(4))
b


# In[71]:


appli_train[b.index].info()


# In[72]:


cols=list(set(appli_train.select_dtypes(include=['float'])))


# In[73]:


appli_train[cols]=appli_train[cols].fillna(appli_train[cols].mean())


# In[74]:


appli_train.shape


# In[75]:


appli_train.head()


# In[76]:


categ_null=appli_train[list(appli_train.select_dtypes(include='object').columns)].isna().sum()
categ_null[categ_null>0]


# In[77]:


numerical_null=appli_train[list(appli_train.select_dtypes(include=['float','int']).columns)].isna().sum()
numerical_null[numerical_null>0]


# # DATA IS CLEAN

# # Working with Target Column

# In[78]:


appli_train['TARGET'].head()


# In[79]:


np.sum(appli_train.TARGET==1)


# In[80]:


np.sum(appli_train.TARGET==0)


# In[81]:


train_0 = appli_train.loc[appli_train.TARGET==0]
train_1 = appli_train.loc[appli_train.TARGET==1]


# In[82]:


train_categ=appli_train.select_dtypes(include='object').columns
train_categ


# In[83]:


def plotting(train, train0, train1, column):
    train = train
    train_0 = train0
    train_1 = train1
    col = column
    
    fig = plt.figure(figsize=(13,10))
    
    ax1=plt.subplot(221)
    train[col].value_counts().plot.pie(autopct = '%1.0f%%', ax=ax1)
    plt.title('Plotting Data for the column: '+ column)
    
    ax2 = plt.subplot(222)
    sns.countplot(x=column, hue='TARGET', data = train, ax = ax2)
    plt.xticks(rotation=90)
    plt.title('Plotting data for target in terms of total count')
    
    ax3 = plt.subplot(223)
    df = pd.DataFrame()
    df['0']=((train_0[col].value_counts())/len(train_0))
    df['1']=((train_1[col].value_counts())/len(train_1))
    df.plot.bar(ax=ax3)
    plt.title('Plotting data for Target in terms of percentage')
    
    fig.tight_layout()
    
    plt.show()


# ### Plotting Pie charts, Bar chart, Count plot of categorical columns. There are 12 categorical columns. We are getting 36 plots.

# In[84]:


for column in train_categ:
    print('Plotting',column)
    plotting(appli_train, train_0, train_1, column)
    print('end of above Column')


# ## NUMERICAL COLUMN

# ### Making heatmap of all numerical columns to see the correlation.

# ### Correlation of columns having Target value equals to 0.

# In[85]:


corr = train_0.corr()
f, ax = plt.subplots(figsize=(11,9))
with sns.axes_style('white'):
    ax = sns.heatmap(corr, vmax=.3, square=True)


# ## Finding Top 10 Correlation

# In[86]:


train_0.corr()


# In[87]:


train_0.corr().abs()


# In[88]:


train_0.corr().abs().unstack()


# In[89]:


train_0.corr().abs().unstack().sort_values()


# In[90]:


correlation_0=train_0.corr().abs().unstack().sort_values().dropna()
correlation_0


# In[91]:


correlation_0 = correlation_0[correlation_0 != 1.0]
correlation_0


# ### Correlation of columns having Target value equals to 1.

# In[92]:


corr = train_1.corr()
f, ax = plt.subplots(figsize=(11,9))
with sns.axes_style('white'):
    ax = sns.heatmap(corr, vmax=.3, square=True)


# In[93]:


correlation_1=train_1.corr().abs()
correlation_1=correlation_1.unstack().sort_values()
correlation_1=correlation_1.dropna()
correlation_1=correlation_1[correlation_1 != 1.0]
print(correlation_1)


# In[94]:


correlation_1.sort_values(ascending=False).head(10)


# In[95]:


train_categ = appli_train.select_dtypes(include=['int','float']).columns


# In[96]:


train_categ


# ## Analysis for Outliers

# Plotting the nemerical based on index and analysing if there are outliers in any of the column

# In[97]:


for column in train_categ:
    title = 'Plot of '+ column
    plt.scatter(appli_train.index, appli_train[column])
    plt.title(title)
    plt.show()


# ## Converting a numerical data to categorical for analysis

# In[98]:


def amt_annuity(x):
    if x < 20000:
        return 'low'
    elif x > 20000 and x<=50000:
        return 'mid'
    elif x>50000 and x<=100000:
        return 'high'
    else:
        return 'v.high'


# In[99]:


appli_train['AMT_ANNUITY_CATEG'] = appli_train['AMT_ANNUITY'].apply(amt_annuity)


# In[100]:


appli_train['AMT_ANNUITY_CATEG'].value_counts()


# In[101]:


label= ['mid','low','high','v.high']
sns.countplot(x = 'AMT_ANNUITY_CATEG', data = appli_train).set_xticklabels(labels=label,rotation=90)
plt.show()


# In[102]:


for column in train_categ:
    title = 'Plot of ' + column
    print(title)
    plt.hist(train_0[column], alpha=0.5, label='0')
    plt.hist(train_1[column], alpha=0.5, label='1')
    
    plt.show()
    print('end of above column')


# # Reading Previous Application

# In[103]:


prev_appli.shape


# In[104]:


prev_appli.head()


# ### There are duplicates 'SK_ID_CURR' as the person could have taken loan multiple times

# In[105]:


prev_appli.SK_ID_PREV.value_counts()


# In[106]:


prev_appli.SK_ID_CURR.value_counts()


# ####   As you can see above, the shape of previous application is (1670214, 37) and the length of SK_ID_PREV is also (1670214), but length of SK_ID_CURR is (338857), which is less than length of SK_ID_PREV, which tells us that there are duplicate number of SK_ID_PREV

# ### Let's merge dataframe: train and previous application based on SK_ID_PREV

# #### After merging both the dataframe, the new dataframe will also have duplicate number of SK_ID_PREV. This should not be a  problem, as we are trying to figure out if any pattern is present by including the cases if lender has previously taken loan more than once.

# In[107]:


prev_train=appli_train.merge(prev_appli, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')


# In[108]:


prev_train.shape


# In[109]:


prev_train.head()


# In[110]:


prev_appli.columns.values


# #### The merged dataframe also has multiple values for SK_ID_CURR

# In[111]:


prev_appli.SK_ID_CURR.value_counts().head()


# #### Segregating the dataframe on Target=0 and Target=1

# In[112]:


train_0 = appli_train.loc[appli_train['TARGET']==0]
train_1 = appli_train.loc[appli_train['TARGET']==1]


# In[113]:


ptrain_0 = appli_train.loc[prev_train['TARGET']==0]
ptrain_1 = appli_train.loc[prev_train['TARGET']==1]


# ### function for plotting data

# In[114]:


def plotting(column, hue):
    col = column
    hue = hue
    fig = plt.figure(figsize=(13,10))

    ax1 = plt.subplot(221)
    appli_train[col].value_counts().plot.pie(autopct = "%1.0f%%", ax=ax1)
    plt.title('Plotting data for the column: '+ column)


    ax2 = plt.subplot(222)
    df = pd.DataFrame()
    df['0']= ((train_0[col].value_counts())/len(train_0))
    df['1']= ((train_1[col].value_counts())/len(train_1))
    df.plot.bar(ax=ax2)
    plt.title('Plotting data for target in terms of total count')


    ax3 = plt.subplot(223)
    sns.countplot(x=col, hue=hue, data=ptrain_0, ax = ax3)
    plt.xticks(rotation=90)
    plt.title('Plotting data for Target=0 in terms of percentage')

    ax4 = plt.subplot(224)
    sns.countplot(x=col, hue=hue, data=ptrain_1, ax = ax4)
    plt.xticks(rotation=90)
    plt.title('Plotting data for Target=1 in terms of percentage')



    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    plt.show()

