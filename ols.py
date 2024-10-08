import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from scipy import stats
import statsmodels.api as sm

#import data
train = pd.read_csv('D:/2023semester/spyder/project1/HWA/housing/train.csv')

#drop irrelevant data
train = train.drop('Id', axis = 1)

#missing values
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(30)#see percentage of missing value(https://www.kaggle.com/code/dhananjaysawarkar/house-price-prediction-for-beginners)

threshold = 0.10
columns_to_drop = train.columns[train.isnull().mean() > threshold]
train = train.drop(columns_to_drop, axis=1)#delete missing value above 10%

cols_to_fix = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish',
'GarageQual', 'GarageCond','GarageYrBlt']
train[cols_to_fix] = train[cols_to_fix]. fillna ('NotAvailable')#deal missing value
train.replace('NotAvailable', np.nan, inplace=True)#deal missing value

#treat the non-numeric variables
    #label encoding
label_encoding_cols = [
    'Street', 'LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 
    'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
    'PavedDrive'
]
label_encoder = LabelEncoder()
for col in label_encoding_cols:
    train[col] = label_encoder.fit_transform(train[col])
    #one-hot encoding
one_hot_encoding_cols = [
    'MSZoning', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 
    'Foundation', 'BsmtFinType2', 'CentralAir', 'SaleType', 'SaleCondition'
]
train = pd.get_dummies(train, columns=one_hot_encoding_cols, prefix=one_hot_encoding_cols)


#reduce dimension
#use lasso
X = train.drop('SalePrice', axis=1)  # Features
y = train['SalePrice']  # Target variable
column_names = X.columns
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 100), cv=5)
lasso_cv.fit(X, y)
optimal_alpha = lasso_cv.alpha_
selected_indices = lasso_cv.coef_ != 0
train = column_names[selected_indices]
train = pd.DataFrame(X[:, selected_indices], columns=train)

#show Y
y.describe()
sns.distplot(y)

#deal outlier X and Y
#oulier X
z_scores = np.abs(stats.zscore(train))
threshold = 3
train = train[(z_scores < threshold).all(axis=1)]
train.to_excel('D:/2023semester/spyder/project1/HWA/train(late).xlsx', index=False)#Z-score
#outlier Y
threshold = 3
y = y[(z_scores < threshold).all(axis=1)]#Z-score
y = np.log(y)#log Y

#match X and y
common_indices = train.index.intersection(y.index)
X_matched = train.loc[common_indices]
y_matched = y.loc[common_indices]
print(train)

#OLS
X_matched = sm.add_constant(X_matched)
model = sm.OLS(y_matched, X_matched).fit()
print(model.summary())





# # Train your final model using the selected features and optimal alpha
# final_model = YourModelHere()  # Replace with your preferred model
# final_model.fit(selected_features, y)










# #delete highly related（PCA）(too many variables in the end)
# train.drop("SalePrice", axis=1, inplace=True)#drop dependent variable
# numeric_columns = train.select_dtypes(include=['number'])#select only number
# scaler = StandardScaler()
# imputer = SimpleImputer(strategy='mean')
# numeric_columns_scaled = scaler.fit_transform(imputer.fit_transform(numeric_columns))
# train_scaled = pd.DataFrame(data=numeric_columns_scaled, columns=numeric_columns.columns)#deal data

# n_components = 0.95
# pca = PCA(n_components=n_components)
# train_pca = pca.fit_transform(train_scaled)#PCA

# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)#see explained variance

# cumulative_variance = np.cumsum(explained_variance)
# target_variance = 0.95
# num_components_to_keep = np.argmax(cumulative_variance >= target_variance) + 1
# train_pca = train_pca[:, :num_components_to_keep]#change data

# train_pca_df = pd.DataFrame(data=train_pca, columns=numeric_columns.columns[:num_components_to_keep])
# updated_column_names = train.columns
# one_hot_encoding_cols = updated_column_names
# train = pd.concat([train_pca_df, train[one_hot_encoding_cols]], axis=1)


# train.to_excel('D:/2023semester/spyder/project1/HWA/111.xlsx')
# print(train)









# ##
# #y=train['SalePrice']
# #x=train['']
# #print(y.shape)
# #print(x)
