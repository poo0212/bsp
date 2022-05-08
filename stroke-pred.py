#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imputing missing values
from sklearn.impute import KNNImputer

from scipy.stats import chi2_contingency

# Feature engineering
from sklearn.preprocessing import StandardScaler

# Model processing and testing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, plot_roc_curve, precision_score, recall_score

# Models

from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("StrokeDS.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


s0 = round(df[df['stroke'] == 0].describe(), 2)
s1 = round(df[df['stroke'] == 1].describe(), 2)

pd.concat([s0, s1], axis = 1, keys = ['No Stroke', 'Stroke'])


# In[7]:


df.isnull().sum()


# In[8]:


def count_negatives(data):
    neg_count = 0
    for n in data:
        if type(data) == 'int':
            if n < 0:
               neg_count += 1
    return neg_count

count_negatives(df)


# In[9]:


df_knn = df.copy()
df_knn.head()


# In[10]:


impute = KNNImputer(n_neighbors = 5, weights = 'uniform')
df_knn['bmi'] = impute.fit_transform(df_knn[['bmi']])


# In[11]:


df_knn.isnull().sum()


# In[12]:


colors = ["#f1d295", "#c8c14f", "#fa8775", "#ea5f94", "#cd34b5", "#9d02d7"]
palette = sns.color_palette(palette = colors)

sns.palplot(palette, size = 2)
plt.text(-0.5, -0.7, 'Color Palette For This Notebook', size = 20, weight = 'bold')


# In[13]:


fig, ax = plt.subplots(figsize = (10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

sns.histplot(
    df['age'],
    kde = False,
    color = "#ea5f94"
)

for i in ['top', 'left', 'bottom', 'right']:
    ax.spines[i].set_visible(False)

plt.text(5, 360, r'$\mu$ = '+str(round(df['age'].mean(), 2)), fontsize = 12)
plt.text(5, 343, r'$\sigma$ = '+str(round(df['age'].std(), 2)), fontsize = 12)
plt.title('Frequency of Ages', fontsize = 18, fontweight = 'bold', pad = 10)
plt.xlabel('Age', fontsize = 14, labelpad = 10)
plt.ylabel('Count', fontsize = 14, labelpad = 10)


# In[14]:


fig, ax = plt.subplots(figsize = (10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

sns.histplot(
    df['avg_glucose_level'],
    color = "#ea5f94",
    kde = False
)

for i in ['top', 'left', 'bottom', 'right']:
    ax.spines[i].set_visible(False)


plt.text(220, 360, r'$\mu$ = '+str(round(df['avg_glucose_level'].mean(), 2)), fontsize = 12)
plt.text(220, 340, r'$\sigma$ = '+str(round(df['avg_glucose_level'].std(), 2)), fontsize = 12)
plt.title('Frequency of Glucose Levels', fontsize = 18, fontweight = 'bold', pad = 10)
plt.xlabel('Average Glucose Level', fontsize = 14, labelpad = 10)
plt.ylabel('Count', fontsize = 14, labelpad = 10)


# In[15]:


fig, ax = plt.subplots(1, 2, figsize = (12, 7))
fig.patch.set_facecolor('#faf9f7')
ax[0].set_facecolor('#faf9f7')
ax[1].set_facecolor('#faf9f7')

sns.histplot(
    df['bmi'],
    color = "#ea5f94",
    kde = False,
    ax = ax[0]
)

sns.histplot(
    df_knn['bmi'],
    color = "#ea5f94",
    kde = False,
    ax = ax[1]
)

ax[0].text(70, 330, r'$\mu$ = '+str(round(df['bmi'].mean(), 2)), fontsize = 11)
ax[0].text(70, 320, r'$\sigma$ = '+str(round(df['bmi'].std(), 2)), fontsize = 11)
ax[0].set_title('Original BMI Data', fontsize = 16, fontweight = 'bold', pad = 10)
ax[0].set_xlabel('BMI', fontsize = 13)
ax[0].set_ylabel('Count', fontsize = 13)

ax[1].text(70, 500, r'$\mu$ = '+str(round(df_knn['bmi'].mean(), 2)), fontsize = 11)
ax[1].text(70, 485, r'$\sigma$ = '+str(round(df_knn['bmi'].std(), 2)), fontsize = 11)
ax[1].set_title('KNN Imputed BMI Data', fontsize = 16, fontweight = 'bold', pad = 10)
ax[1].set_xlabel('BMI', fontsize = 13)
ax[1].set_ylabel('')

for i in ['top', 'left', 'bottom', 'right']:
    ax[0].spines[i].set_visible(False)
    ax[1].spines[i].set_visible(False)


plt.tight_layout()


# In[16]:


df['bmi'] = df_knn['bmi']
df['bmi'].isnull().sum()


# In[17]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (14,6))
fig.patch.set_facecolor('#faf9f7')

for i in (ax1, ax2, ax3):
    i.set_facecolor('#faf9f7')

sns.kdeplot(
    df['age'][df['stroke'] == 0],
    ax = ax1,
    color = "#c8c14f",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

sns.kdeplot(
    df['age'][df['stroke'] == 1],
    ax = ax1,
    color = "#cd34b5",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)
ax1.legend(['No Stroke', 'Stroke'], loc = 'upper left')
ax1.set_xlabel('Age', fontsize = 14, labelpad = 10)
ax1.set_ylabel('Density', fontsize = 14, labelpad = 10)

sns.kdeplot(
    df['avg_glucose_level'][df['stroke'] == 0],
    ax = ax2,
    color = "#c8c14f",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

sns.kdeplot(
    df['avg_glucose_level'][df['stroke'] == 1],
    ax = ax2,
    color = "#cd34b5",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

ax2.legend(['No Stroke', 'Stroke'])
ax2.set_xlabel('Average Glucose Levels', fontsize = 14, labelpad = 10)
ax2.set_ylabel('')

sns.kdeplot(
    df['bmi'][df['stroke'] == 0],
    ax = ax3,
    color = "#c8c14f",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

sns.kdeplot(
    df['bmi'][df['stroke'] == 1],
    ax = ax3,
    color = "#cd34b5",
    shade = True,
    alpha = 0.5,
    linewidth = 1.5,
    ec = 'black'
)

ax3.legend(['No Stroke', 'Stroke'])
ax3.set_xlabel('BMI', fontsize = 14, labelpad = 10)
ax3.set_ylabel('')

plt.suptitle('Density of Age, Glucose, and BMI by Stroke', fontsize = 16, fontweight = 'bold')

for i in (ax1, ax2, ax3):
    for j in ['top', 'left', 'bottom', 'right']:
        i.spines[j].set_visible(False)

fig.tight_layout()


# In[18]:


stroke = df[df['stroke'] == 1]
no_stroke = df[df['stroke'] == 0]


# In[19]:


fig, ax = plt.subplots(3, 1, figsize=(16,20))
fig.patch.set_facecolor('#faf9f7')
for j in range(0, 3):
    ax[j].set_facecolor('#faf9f7')

## Age vs Glucose Levels
sns.scatterplot(
    data = no_stroke, x = 'age', y = 'avg_glucose_level', color = '#f1d295',
    alpha = 0.4, ax = ax[0]
)
sns.scatterplot(
    data = stroke, x = 'age', y = 'avg_glucose_level', color = "#9d02d7",
    ax = ax[0], edgecolor = 'black', linewidth = 1.2, alpha = 0.6
)

# Age vs BMI
sns.scatterplot(
    data = no_stroke, x = 'age', y = 'bmi', color = '#f1d295',
    alpha = 0.4, ax = ax[1]
)
sns.scatterplot(
    data = stroke, x = 'age', y = 'bmi', color = "#9d02d7",
    ax = ax[1], edgecolor = 'black', linewidth = 1.2, alpha = 0.6
)

# Glucose Levels vs BMI
sns.scatterplot(
    data = no_stroke, x = 'avg_glucose_level', y = 'bmi', color = '#f1d295',
    alpha = 0.4, ax = ax[2]
)
sns.scatterplot(
    data = stroke, x = 'avg_glucose_level', y = 'bmi', color = "#9d02d7",
    ax = ax[2], edgecolor = 'black', linewidth = 1.2, alpha = 0.6
)
    
sns.despine()

for i in range(0, 3, 1):
    ax[i].legend(['No Stroke', 'Stroke'])

fig.tight_layout()


# In[20]:


fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

labels = ['No Stroke', 'Stroke']
colors = ["#f1d295", "#ea5f94"]
sizes = df['stroke'].value_counts()

plt.pie(sizes, explode = [0, 0.15], labels = labels, colors = colors,
           autopct = '%1.1f%%', shadow = True, startangle = 130,
           wedgeprops = {'ec': 'black'}, textprops = {'fontweight': 'medium'}
        
)

plt.axis('equal')
plt.title('Percentage of Strokes')


# In[21]:


male_str = 0
fem_str = 0
male_nstr = 0
fem_nstr = 0

for index, row in df.iterrows():
    if row['gender'] == 'Male':
        if row['stroke'] == 1:
            male_str += 1
        else:
            male_nstr += 1
    else:
        if row['stroke'] == 1:
            fem_str += 1
        else:
            fem_nstr += 1

print(male_str, fem_str, male_nstr, fem_nstr)


# In[22]:


plt.subplots(figsize=(8,6))

stroke_matrix = np.array([[108, 2007], [141, 2854]])
labels = np.array([['Male - Stroke', 'Male - No Stroke'], ['Female - Stroke', 'Female - No Stroke']])
formatted = (np.asarray(["{0}\n{1:.0f}".format(text, data) for text, data in zip(labels.flatten(), stroke_matrix.flatten())])).reshape(2,2)


sns.heatmap(
    stroke_matrix,
    annot = formatted,
    fmt = '',
    cmap = palette,
    xticklabels = False,
    yticklabels = False,
    linecolor = 'black',
    linewidth = 1,
    annot_kws = {'fontweight': 'semibold'}
)
plt.title('Two-Way Contingency Table of Strokes by Gender', pad = 15, fontsize = 14)
plt.ylabel('Gender', fontsize = 12, labelpad = 10)
plt.xlabel('Stroke', fontsize = 12, labelpad = 10)


# In[23]:


heart_cont = pd.crosstab(df['heart_disease'], df['stroke'])
heart_cont


# In[24]:


plt.subplots(figsize=(8,6))

heart_matrix = np.array([[3636, 941], [154, 498]])
labels = np.array([['No Heart Disease - No Stroke', 'No Heart Disease - Stroke'], ['Heart Disease - No Stroke', 'Heart Disease - Stroke']])
formatted = (np.asarray(["{0}\n{1:.0f}".format(text, data) for text, data in zip(labels.flatten(), heart_matrix.flatten())])).reshape(2,2)

sns.heatmap(
    heart_cont,
    annot = formatted,
    fmt = '',
    cmap = palette,
    linewidth = 1,
    linecolor = 'black',
    xticklabels = False,
    yticklabels = False,
    annot_kws = {'fontweight': 'semibold'}
)
plt.ylabel('Heart Disease', labelpad = 10, fontsize = 12)
plt.xlabel('Stroke', labelpad = 10, fontsize = 12)


# In[25]:


stat, p, dof, expected = chi2_contingency(heart_cont)
stat, p


# In[26]:


hyper_cont = pd.crosstab(df['hypertension'], df['stroke'])
hyper_cont


# In[27]:


plt.subplots(figsize=(8,6))

hyper_matrix = np.array([[3527, 717], [263, 722]])
labels = np.array([['No Hypertension - No Stroke', 'No Hypertension - Stroke'], ['Hypertension - No Stroke', 'Hypertension - Stroke']])
formatted = (np.asarray(["{0}\n{1:.0f}".format(text, data) for text, data in zip(labels.flatten(), hyper_matrix.flatten())])).reshape(2,2)

sns.heatmap(
    hyper_cont,
    annot = formatted,
    fmt = '',
    cmap = palette,
    linewidth = 1,
    linecolor = 'black',
    xticklabels = False,
    yticklabels = False,
    annot_kws = {'fontweight': 'semibold'}
)
plt.ylabel('Hypertension', labelpad = 10, fontsize = 12)
plt.xlabel('Stroke', labelpad = 10, fontsize = 12)


# In[28]:


df.groupby('Residence_type')['stroke'].value_counts()


# In[29]:


res_cont = pd.crosstab(df['Residence_type'], df['stroke'])
res_cont


# In[30]:


plt.subplots(figsize=(8,6))

res_matrix = np.array([[1760, 714], [1884, 734]])
labels = np.array([['Rural - No Stroke', 'Rural - Stroke'], ['Urban - No Stroke', 'Urban - Stroke']])
formatted = (np.asarray(["{0}\n{1:.0f}".format(text, data) for text, data in zip(labels.flatten(), res_matrix.flatten())])).reshape(2,2)

sns.heatmap(
    res_cont,
    annot = formatted,
    fmt = '',
    cmap = palette,
    linewidth = 1,
    linecolor = 'black',
    xticklabels = False,
    yticklabels = False,
    annot_kws = {'fontweight': 'semibold'}
)
plt.ylabel('Residence Type', labelpad = 10, fontsize = 12)
plt.xlabel('Stroke', labelpad = 10, fontsize = 12)


# In[31]:


mar_cont = pd.crosstab(df['ever_married'], df['stroke'])
mar_cont


# In[32]:


plt.subplots(figsize=(8,6))

mar_matrix = np.array([[1728, 29], [3133, 220]])
labels = np.array([['Never Married - No Stroke', 'Never Married - Stroke'], ['Married - No Stroke', 'Married - Stroke']])
formatted = (np.asarray(["{0}\n{1:.0f}".format(text, data) for text, data in zip(labels.flatten(), mar_matrix.flatten())])).reshape(2,2)

sns.heatmap(
    mar_cont,
    annot = formatted,
    fmt = '',
    cmap = palette,
    linewidth = 1,
    linecolor = 'black',
    xticklabels = False,
    yticklabels = False,
    annot_kws = {'fontweight': 'semibold'}
)
plt.ylabel('Ever Married', labelpad = 10, fontsize = 12)
plt.xlabel('Stroke', labelpad = 10, fontsize = 12)


# In[33]:


df['smoking_status'].unique()


# In[34]:


df.groupby('smoking_status')['stroke'].value_counts()


# In[35]:


fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

bar_pal = ["#c8c14f", "#fa8775"]

s = sns.countplot(
    data = df, x = 'smoking_status', hue = 'stroke', palette = bar_pal,
    linewidth = 1.2, ec = 'black'
)

for i in ['top', 'right', 'bottom', 'left']:
    ax.spines[i].set_visible(False)

plt.legend(['No Stroke', 'Stroke'])
plt.title("Smoking Status' Effect on Stroke", size = 16, weight = 'bold', pad = 12)
plt.xlabel('Smoking Status', size = 12, labelpad = 12)
plt.ylabel('Count', size = 12, labelpad = 12)

for i in s.patches:
    s.annotate(format(i.get_height(), '.0f'),  (i.get_x() + i.get_width() / 2., i.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

fig.tight_layout()


# In[36]:


df['work_type'].unique()


# In[37]:


fig, ax = plt.subplots(figsize=(10,6))
fig.patch.set_facecolor('#faf9f7')
ax.set_facecolor('#faf9f7')

bar_pal = ["#c8c14f", "#fa8775"]

w = sns.countplot(
    data = df, x = 'work_type', hue = 'stroke', palette = bar_pal,
    linewidth = 1.2, ec = 'black'
)

for i in ['top', 'right', 'bottom', 'left']:
    ax.spines[i].set_visible(False)

plt.legend(['No Stroke', 'Stroke'])
plt.title("Work Type's Effect on Stroke", size = 16, weight = 'bold', pad = 12)
plt.xlabel('Work Type', size = 12, labelpad = 12)
plt.ylabel('Count', size = 12, labelpad = 12)

for i in w.patches:
    w.annotate(format(i.get_height(), '.0f'),  (i.get_x() + i.get_width() / 2., i.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

fig.tight_layout()


# In[38]:


gen_odds = (108 * 2854) / (141 * 2007)

heart_odds = (229 * 202) / (4632 * 47)

hyper_odds = (432 * 183) / (4429 * 66)

res_odds = (2400 * 135) / (2461 * 114)

mar_odds = (1728 * 220) / (3133 * 29)

d = {
    'Features': ['Gender', 'Heart Disease', 'Hypertension',
                'Residence', 'Married'],
    'Odds': [gen_odds, heart_odds, hyper_odds, res_odds, mar_odds]
}

odds_df = pd.DataFrame(data = d)
odds_df


# In[39]:


df = pd.get_dummies(df, columns = ['gender', 'work_type', 'Residence_type', 'smoking_status'], prefix = ['sex', 'work', 'residence', 'smoke'])
df.head()


# In[40]:


df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
df.head()


# In[41]:


#num_cols = ['age', 'avg_glucose_level', 'bmi']

#scaler = StandardScaler()

#df[num_cols] = scaler.fit_transform(df[num_cols])
num_cols=df.select_dtypes(include=['object']).columns
print(num_cols)
# This code will fetech columns whose data type is object.
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
# Initializing our Label Encoder object
df[num_cols]=df[num_cols].apply(le.fit_transform)
# Transfering categorical data into numeric
print(df.head(10))


# In[ ]:





# In[42]:


df.head()


# In[ ]:





# In[43]:


df = df.drop('id', axis = 1)
df.head()


# In[44]:


df.isnull().sum()


# x = df.drop('stroke', axis = 1)
# y = df['stroke']
# 
# #from imblearn.over_sampling import RandomOverSampler
# #ros = RandomOverSampler(random_state=42)# fit predictor and target variable
# #x_ros, y_ros = ros.fit_resample(x, y)
# #smote = SMOTE()
# 
# #x_oversample, y_oversample = smote.fit_resample(x, y)
# 
# 
# from imblearn.over_sampling import SMOTE
# 
# smote = SMOTE()
# 
# #fit predictor and target variable
# x_smote, y_smote = smote.fit_resample(x, y)
# print(y.value_counts())
# print(y_smote.value_counts())
# 

# In[45]:


df.isnull().sum()


# In[46]:


x = df.drop('stroke', axis = 1)
y = df['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# #from imblearn.over_sampling import SMOTE
# #smote = SMOTE()
# #x_smote, y_smote = smote.fit_resample(x, y)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# 
# cr_rf = classification_report(y_test, y_pred)
# print(cr_rf)

# In[47]:


#Randpm forest model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
cr_rf = classification_report(y_test, y_pred_rf)
print(cr_rf)

#ranfor_model = RandomForestClassifier(n_estimators=100, random_state=42)
#ranfor_model.fit(x_train, y_train)


# Predicting the test set results
#y_ranfor = ranfor_model.predict(x_test)
#y_ranfor_prob = ranfor_model.predict_proba(x_test)   
#print("Classification report for RF: \n{}".format(classification_report(y_test,y_ranfor)))


# print('Precision Score: ', round(precision_score(y_test, y_ranfor_prob), 2))
# print('Recall Score: ', round(recall_score(y_test, y_ranfor_prob), 2))
# print('F1 Score: ', round(f1_score(y_test, y_ranfor_prob), 2))
# print('Accuracy Score: ', round(accuracy_score(y_test, y_ranfor_prob), 2))
# print('ROC AUC: ', round(roc_auc_score(y_test, y_ranfor_prob), 2))

# In[48]:


plot_roc_curve(rf, x_test, y_test)


# In[49]:


sns.heatmap(
    confusion_matrix(y_test, y_pred_rf),
    cmap = palette,
    annot = True,
    fmt = 'd',
    yticklabels = ['No Stroke', 'Stroke'],
    xticklabels = ['Pred No Stroke', 'Pred Stroke']
)


# In[50]:


#input_data = (67.0,0,1,1,228.69,36.600000,1,0,1,0,1,0,0,0,1,0,1,0,0) #stroke
input_data=(58.0,1,0,1,223.36,41.5,0,0,1,0,1,0,0,1,0,0,1,0,0) #NoStroke
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rf.predict(input_data_reshaped)
print(prediction)

if (prediction == 0):
  print('The person doesnt have any chance of stroke')
else:
  print('The person have high chance of stroke')


# In[51]:


import pickle
with open(r'C:\Poonav\models\model.sav','wb')as file:
    pickle.dump(rf,file)


# 
# # pandas profiling
# from pandas_profiling import ProfileReport
# profile=ProfileReport(df)
# profile.to_file(output_file= r"E:\MscIt proj\stroke\templates\stroke.html")

# In[52]:


input_data = (67.0,0,1,1,228.69,36.600000,1,0,1,0,1,0,0,0,1,0,1,0,0) #stroke
#input_data=(58.0,1,0,1,223.36,41.5,0,0,1,0,1,0,0,1,0,0,1,0,0) #NoStroke
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = rf.predict(input_data_reshaped)
print(prediction)

if (prediction == 0):
  print('The person doesnt have any chance of stroke')
else:
  print('The person have high chance of stroke')


# In[ ]:




