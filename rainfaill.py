import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report


# Suppressing warnings


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r'/home/mucha/Documents/ML Projects/Weather Prediction/rainfall/Rainfall.csv')

# Exploring the dataset

print(data.head())
print(data.info())
print(data.describe().T)

# Checking for missing values  

print(data.isnull().sum())

# Removing unnecessary spaces in column names

data.columns = data.columns.str.replace(' ', '')
print(data.info())

# Filling missing values with the mean of the column
data['winddirection'] = data['winddirection'].fillna(data['winddirection'].mean())
data['windspeed'] = data['windspeed'].fillna(data['windspeed'].mean())

print(data.isnull().sum())

# Cheking for the days which has rained when oberving the data

print(data['rainfall'].value_counts())
plt.pie(data['rainfall'].value_counts(), labels=['Rain', 'No Rain'], autopct='%1.1f%%')
plt.title('Rainfall Distribution')
plt.show()

#cheking for the asscocian/ relationship between the freatures and the rainfall

features = data.drop(['day','rainfall'], axis=1)
print(features.head())

plt.subplots(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sns.histplot(data[col], kde=True)
plt.tight_layout()
plt.show()

# Getting the boxplots for the features

plt.subplots(figsize=(12, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sns.boxplot(data[col], orient='h')
plt.tight_layout()
plt.show()

#due to lack of the data we have to keep the outling data in the dataset

#cheking the correlation between the features
corr = data.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Dropping the 'day' column as it is not needed for the model
# Also dropping the minimum temperature and maximum temperature columns as they are highly correlated with the mean temperature
# and setting the target variable as 'rainfall'

features = data.drop(['day', 'rainfall' , 'mintemp', 'maxtemp'], axis=1)
target = data['rainfall']

target = target.str.strip().str.lower()
print(target.unique())
target = target.map({'no': 0, 'yes': 1})  # Mapping 'No' to 0 and 'Yes' to 1
print(target.head())

# Splitting the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Balacing the dataset by adding data as repetitive rows of minority class

ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
x_train,y_train = ros.fit_resample(x_train, y_train)
print(x_train.head(), y_train.head())
print(x_train.dtypes, y_train.dtypes)



# normalizing the features for stable and faster convergence
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)


# now train them with state-of-art models for classification
# Logistic Regression, XGBoost, SVM


models = [LogisticRegression(), XGBClassifier(), SVC(probability=True)]

for i in range(3):
    models[i].fit(x_train,y_train)

    print(f'{models[i]}: ')
    train_preds = models[i].predict_proba(x_train)
    print('Traning accuracy: ', metrics.roc_auc_score(y_train, train_preds[:,1]))

    test_preds = models[i].predict_proba(x_test)
    print('Testing accuracy: ', metrics.roc_auc_score(y_test, test_preds[:,1]))
    print()


# Model evaluation

print(metrics.classification_report(y_test, models[2].predict(x_test), target_names=['No Rain', 'Rain']))
