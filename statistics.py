#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for better and easier plots
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
#import scikitplot as skplt
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score,recall_score, f1_score
from pprint import pprint
import random


# In[75]:


df = pd.read_excel(r'C:\Users\Bar\Desktop\features_grapewine.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df.head(5)
df1=df.drop(['Y'], axis=1)
df1.info()


# In[43]:


description=df.describe(include='all')  
description


# In[101]:


sns.countplot(data=df, x="Y")
plt.show()
s = df['Y'].value_counts()
print(s)


# # Histograms and box-plots

# In[45]:


for i in df1.columns:

# sns.set(rc={'figure.figsize':(15,8.7)}) #setting the size of the figure to make it easier to read.
    sns.displot(data=df, x=i, col="Y", kde=True)


# In[70]:


for i in df1.columns:
    sns.catplot(x="Y", y=i, data=df,orient="v", palette="Set2",linewidth=2.5, kind="box")


# In[3]:


#data without CWSI4

df=df.drop(['CWSI4'], axis=1)
df1=df.drop(['Y'], axis=1)

df_outliters=pd.read_excel(r'C:\Users\Bar\Desktop\normalized_by_severity.xlsx') 


# # Outliers

# In[110]:


df1.boxplot(figsize=(15,5)) 
plt.ylim(-8, 17)
df1.info()


# In[113]:


for i in df1.columns:
    
    Q1 = np.quantile(df1[i],0.25)
    Q3 = np.quantile(df1[i],0.75)
    IQR = Q3 - Q1
    lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
    df_outliters = df_outliters.drop(df_outliters[df_outliters[i] < lower].index)
    df_outliters = df_outliters.drop(df_outliters[df_outliters[i] > upper].index)
df_outliters.to_excel("data_without_outlires.xlsx") 


# In[4]:


df_out=pd.read_excel(r'C:\Users\Bar\Desktop\features_without_outlires.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
sns.countplot(data=df_out, x="Y")
plt.show()
s = df_out['Y'].value_counts()
print(s)


# In[4]:


df_out1=df_out.drop(['Y'], axis=1)
df_out1.dropna(how='any')
df_out1.boxplot(figsize=(15,5)) 
plt.ylim(-8, 17)
df_out1.info()


# # Correlation

# In[5]:


fix,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_out1.corr(),vmax=1,linewidths=0.01,
            square=True,annot=True,linecolor="white", cmap='Reds')
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.show()


# In[6]:


fix,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_out.pcorr(),vmax=1,linewidths=0.01,
            square=True,annot=True,linecolor="white", cmap='Reds')
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.show()


# In[8]:


# Data with selected features

df_data=df_out.drop(['IQR','MAD','Tavg-Tair','Tmin-Tair','Tmax-Tair','median-Tair','perc10-Tair'], axis=1)
#df_data.info()
df_features=df_data.drop(['Y'], axis=1)
#df_features.info()
df_Y=df_data.drop(['MTD','STD','Cv','perc90-Tair','CWSI2'], axis=1)
#df_Y.info()


# In[50]:


df_days=pd.read_excel(r'C:\Users\Bar\Desktop\features_without_outlires_by_days.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
#df_days.info()
df_days_data=df_days.drop(['IQR','MAD','Tavg-Tair','Tmin-Tair','Tmax-Tair','median-Tair','perc10-Tair'], axis=1)
df_days_features=df_days_data.drop(['Y', 'day_after_infection'], axis=1)


# In[26]:


for i in df_days_features.columns:

    sns.displot(data=df_days_data, x=i, col="day_after_infection", kde=True)


# In[25]:


for i in df_days_features.columns:

    sns.displot(data=df_days_data, x=i, col="day_after_infection",stat="density", common_norm=False,kde=True)


# In[64]:


df_days_data_4=df_days_data['day_after_infection']==4
day_4 = df_days_data[df_days_data_4]
print(day_4)
df_day4_features=day_4.drop(['Y', 'day_after_infection'], axis=1)
for i in df_day4_features.columns:

    sns.displot(data=day_4, x=i, col="day_after_infection", kde=True)


# In[86]:


df_day4_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)
df_day4_features.info()


# In[87]:


df_days_data_1=df_days_data['day_after_infection']==1
day_1 = df_days_data[df_days_data_1]
df_day1_features=day_1.drop(['Y', 'day_after_infection'], axis=1)
df_day1_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[82]:


df_days_data_2=df_days_data['day_after_infection']==2
day_2 = df_days_data[df_days_data_2]
df_day2_features=day_2.drop(['Y', 'day_after_infection'], axis=1)
df_day2_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[83]:


df_days_data_5=df_days_data['day_after_infection']==5
day_5 = df_days_data[df_days_data_5]
df_day5_features=day_5.drop(['Y', 'day_after_infection'], axis=1)
df_day5_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[84]:


df_days_data_6=df_days_data['day_after_infection']==6
day_6 = df_days_data[df_days_data_6]
df_day6_features=day_6.drop(['Y', 'day_after_infection'], axis=1)
df_day6_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[85]:


df_days_data_7=df_days_data['day_after_infection']==7
day_7 = df_days_data[df_days_data_7]
df_day7_features=day_7.drop(['Y', 'day_after_infection'], axis=1)
df_day7_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[88]:


df_days_data_0=df_days_data['day_after_infection']==0
day_0 = df_days_data[df_days_data_0]
df_day0_features=day_0.drop(['Y', 'day_after_infection'], axis=1)
df_day0_features.boxplot(figsize=(10,5)) 
plt.ylim(-5, 13)


# In[12]:


df_group=pd.read_excel(r'C:\Users\Bar\Desktop\Data_27_10_2020_daily.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'

df_group_data=df_group.drop(['day_after_infection','plot_num','image_name','IQR','MAD','Tavg-Tair','Tmin-Tair','Tmax-Tair','median-Tair','perc10-Tair','perc75-Tair','perc2-Tair','perc25-Tair','perc98-Tair'], axis=1)
df_group_features=df_group_data.drop(['Y', 'group'], axis=1)
df_group_features.info()


# In[13]:


for i in df_group_features.columns:

    sns.displot(data=df_group_data, x=i, col="group", kde=True)


# In[46]:


for i in df_group_features.columns:
    sns.histplot(data=df_group_data, x=i, kde=True, hue="group")


# In[36]:



for j in df_days_features.columns:
    # Create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(10, 20))
    for column in enumerate(df_days_data['day_after_infection']):
        sns.histplot(df_days_data, x=j)


# In[39]:


# libraries & dataset
import seaborn as sns
import matplotlib.pyplot as plt

for i in df_days_features.columns:
    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
    sns.set(style="darkgrid")

    fig, axs = plt.subplots(1, 4, figsize=(7, 7))
    
    sns.histplot(data=df_days_data, x=i, kde=True, hue="day_after_infection",ax=axs[0, 0])
    sns.histplot(data=df_days_data, x=i, kde=True, ax=axs[0, 0],hue="day_after_infection")
    sns.histplot(data=df_days_data, x=i, kde=True,  ax=axs[0, 1])
    sns.histplot(data=df_days_data, x=i, kde=True,  ax=axs[0, 2])
    sns.histplot(data=df_days_data, x=i, kde=True,  ax=axs[0, 3])

    plt.show()


# # RF

# In[5]:


# Create a based model

rf = RandomForestClassifier(random_state=0)
rf.fit(df_features, df_Y)
y_pred = cross_val_predict(rf, df_features, df_Y, cv=10)


print("acc: \n",accuracy_score(df_Y, y_pred))
print("pres: \n",precision_score(df_Y, y_pred))
print("recall: \n",recall_score(df_Y, y_pred))


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
# pprint(rf.get_params())


# # grid search

# In[19]:


from sklearn.model_selection import GridSearchCV

random.seed(123)

# Create the parameter grid based on the results of random search
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [80, 90, 100, 110],
#    'max_features': [2, 3, 4],
#    'min_samples_leaf': [2, 3, 4, 5],
#    'min_samples_split': [8 , 10, 12, 14],
#    'n_estimators': [100, 200, 300, 1000]
#}

param_grid = {
    'bootstrap': [True],
    'max_depth': [80],
    'max_features': [2],
    'min_samples_leaf': [5],
    'min_samples_split': [14],
    'n_estimators': [1000]
}

rf = RandomForestClassifier(random_state=0)
#rf.fit(df_features, df_Y)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(df_features, df_Y)
best_grid=grid_search.best_estimator_
print(" ")
print('best_grid:\n')
print(best_grid)
y_pred_grid=best_grid.predict(df_features)
#print(best_grid.predict(df_features))
print(" ")
print('Parameters after gridsearch in use:\n')

grid_search.best_params_


# In[22]:


scores = cross_val_score(best_grid, df_features, df_Y, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[15]:


print("acc: \n",accuracy_score(df_Y, y_pred_grid))
print("F1 score: \n",f1_score(df_Y, y_pred_grid))
print("precision: \n",precision_score(df_Y, y_pred_grid))
print("recall: \n",recall_score(df_Y, y_pred_grid))


# In[ ]:





# In[17]:


# Plot confusion matrix for base model with crossvalidation
cnf = confusion_matrix(df_Y, y_pred)
plt.figure()
#skplt.metrics.plot_confusion_matrix(df_Y, y_pred, normalize=False)
#plt.show()


# In[18]:


# Plot feature importance
feature_importance = rf.feature_importances_
# make importances relative to max importance
plt.figure(figsize=(10, 10))
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, df_out1.columns[sorted_idx], fontsize=30)
plt.xlabel('Relative Importance', fontsize=30)
plt.title('variable importance', fontsize=40)


# In[36]:


sns.pairplot(data=df, hue="Y")


# correlation Y and the rest

# split the data if wished

# In[20]:


from sklearn.model_selection import train_test_split

X = df.drop('Y', 1).values
Y = df['Y'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42)#

X_train=pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)
X_test=pd.DataFrame(X_test)
y_test=pd.DataFrame(y_test)

print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape) # let's check the shape of the datasets created


# # GLM

# In[40]:


from scipy import stats
import statsmodels.api as sm

glm_binom = sm.GLM(Y, X, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())


# # data processing

# In[10]:


dataset = pd.read_excel(r'C:\Users\amirc\Desktop\grapevine_statistic.xlsx')
dataset.head()

X = dataset.drop('Y', axis=1)
Y = dataset["Y"]


# In[11]:


X.columns


# In[ ]:





# In[14]:


# X.to_excel(r'C:\Users\amirc\Desktop\DATA.xlsx', index = False)


# In[36]:


sns.pairplot(data=df, hue="Y")


# In[25]:


X = X.drop(['MAD','STD'], axis=1)


# # SVM

# In[6]:




for i in X.columns:
    
    Q1 = np.quantile(X[i],0.25)
    Q3 = np.quantile(X[i],0.75)
    IQR = Q3 - Q1
    lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
    X[i] = np.where(X[i] <lower, lower,X[i])
    X[i] = np.where(X[i] >=upper, upper,X[i])
X.dropna(how='any')
# X.boxplot() 


# In[19]:


amir=X.drop(['MTD', 'STD', 'IQR', 'MAD'],axis=1)
lor=X.drop(['Tmin-Tair', 'Tmax-Tair','median-Tair', 'perc10-Tair', 'perc90-Tair', 'Tavg-Tair'],axis=1)
amir.info()
lor


# In[20]:


amir.boxplot() 


# In[21]:


lor.boxplot()


# In[4]:



def remove_less_significant_features(X, Y):
    sl = 0.5
    regression_ols = None
    columns_dropped = np.array([])
    
    regression_ols = sm.OLS(Y, X).fit()
    max_col = regression_ols.pvalues.idxmax()
    max_val = regression_ols.pvalues.max()
    if max_val > sl:
        X.drop(max_col, axis='columns', inplace=True)
        columns_dropped = np.append(columns_dropped, [max_col])
    return columns_dropped


# In[18]:


X.drop(['MTD','median-Tair','Tmin'],axis=1)


# In[9]:


print(remove_less_significant_features(X, Y))
X.columns


# In[7]:


import random
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
import statsmodels.api as sm  # for finding the p-value
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm

random.seed(10)
clf = svm.SVC(kernel='rbf', C=10, random_state=42)
clf.fit(df_features, df_Y)
y_pred_SVM = cross_val_predict(clf, df_features, df_Y, cv=10)

print("acc: \n",accuracy_score(df_Y, y_pred_SVM))
print("pres: \n",precision_score(df_Y, y_pred_SVM))
print("recall: \n",recall_score(df_Y, y_pred_SVM))


# In[40]:


best = svm.SVC(kernel='rbf', C=10,gamma=1e-2 ,random_state=42)
best.fit(df_features, df_Y)
y_pred_best = cross_val_predict(best, df_features, df_Y, cv=10)
print("acc: \n",accuracy_score(df_Y, y_pred_best))
print("pres: \n",precision_score(df_Y, y_pred_best))
print("recall: \n",recall_score(df_Y, y_pred_best))


# In[21]:


# Plot confusion matrix for base model with crossvalidation
cnf = confusion_matrix(Y, y_pred_best)
plt.figure()
skplt.metrics.plot_confusion_matrix(Y, y_pred_best, normalize=False)
plt.show()


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# ## svm grid search

# In[8]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': [1, 0.1, 0.01, 0.001],
    'C': [0.1, 1, 10, 100],
    'degree':[1,2,3,4]
    }

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = svm.SVC(), param_grid = param_grid,
                          cv = 10, verbose = 2)

# Fit the grid search to the data
grid_search.fit(df_features, df_Y)
grid_search.best_params_
best_grid=grid_search.best_estimator_
y_pred_grid=best_grid.predict(df_features)
print(" ")
print('Parameters after gridsearch in use:\n')

grid_search.best_params_


# In[11]:





# In[12]:


print("acc: \n",accuracy_score(df_Y, y_pred_grid))
print("pres: \n",precision_score(df_Y, y_pred_grid))
print("recall: \n",recall_score(df_Y, y_pred_grid))

