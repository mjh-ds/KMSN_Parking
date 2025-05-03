import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('/workspaces/data/msn_parking/output/commute_clean.csv')

##Daily data

df2 = (   df
    .assign(
        datetime = lambda x: pd.to_datetime(x['datetime']),
        cars = lambda x: x['total_spots'] - x['available_spots'],
        datetime_h = lambda x: x['datetime'].dt.floor('h'),
        datetime_d = lambda x: x['datetime'].dt.floor('d'),

        day_of_week = lambda x: x['datetime'].dt.dayofweek,
        day = lambda x: x['datetime'].dt.day,
        hour = lambda x: x['datetime'].dt.hour,
        minute = lambda x: x['datetime'].dt.minute,
        weekend = lambda x: np.where(x['day_of_week'] >= 4, 1, 0),
    )
)



df_d = (df2
        .groupby(['datetime_d','day_of_week','day','weekend'])['cars']
        .mean()
        .reset_index()
        .assign(diff7 = lambda x: x['cars'].diff(7))
       )

# Box plots for day of the week and day of the month
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.violinplot(x='day_of_week', y='cars', data=df_d, ax=axes[0,0])
sns.violinplot(x='day', y='cars', data=df_d, ax=axes[0,1])
sns.lineplot(x='datetime_d', y='cars', data=df_d, ax=axes[1,0])
sns.violinplot(x='weekend', y='cars', data=df_d, ax=axes[1,1])




##################
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

split_index = int(len(df_d) * 0.95) # Split the DataFrame 
train_df = df_d.iloc[:split_index]

# Define features and target
X = train_df[['day_of_week','day','weekend']]
y = train_df['cars']


ts_cv = TimeSeriesSplit(
    n_splits=10
)

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [300],
    'max_depth': [1,2,3,4,5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3,4,5,6,7,8]
}


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=ts_cv, n_jobs=-1, 
                           verbose=2, scoring='neg_mean_squared_error')
# Fit GridSearchCV on the training data
grid_search.fit(X, y)
# Get the best model from grid search
best_rf = grid_search.best_estimator_
# Predict on the test set

df_d['pred'] = best_rf.predict(df_d[['day_of_week','day','weekend']])
pData = df_d.iloc[split_index:]
sns.lineplot(data = pData,x = "datetime_d", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_d', y='pred', label='Predicted', linestyle="--")

pData = df_d
sns.lineplot(data = pData,x = "datetime_d", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_d', y='pred', label='Predicted', linestyle="--")


mse = mean_squared_error(pData['cars'],pData['pred'])
print(f'Mean Squared Error: {mse}')



























df_d = df.groupby('datetime_d')['cars'].mean().reset_index()
df_d['day_of_week'] = df_d['datetime_d'].dt.dayofweek  # Monday=0, Sunday=6
df_d['day'] = df_d['datetime_d'].dt.day
df_d['weekend'] = np.where(df_d['day_of_week'] >= 5, 1, 0)


# Box plots for day of the week and day of the month
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.boxplot(x='day_of_week', y='cars', data=df_d, ax=axes[0,0])
sns.violinplot(x='day', y='cars', data=df_d, ax=axes[0,1])
sns.lineplot(x='datetime_d', y='cars', data=df_d, ax=axes[1,0])
sns.violinplot(x='weekend', y='cars', data=df_d, ax=axes[1,1])









# Box plots for day of the week and day of the month
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.violinplot(x='day_of_week', y='cars', data=df, ax=axes[0,0])
sns.violinplot(x='day', y='cars', data=df, ax=axes[0,1])
sns.violinplot(x='hour', y='cars', data=df, ax=axes[1,0])
sns.violinplot(x='minute', y='cars', data=df, ax=axes[1,1])
axes[0,0].set_title('Number of Cars by Day of the Week')
axes[0].set_xlabel('Day of the Week')
axes[0].set_ylabel('Number of Cars')
axes[1].set_title('Number of Cars by Day of the Month')
axes[1].set_xlabel('Day of the Month')
axes[1].set_ylabel('Number of Cars')








df2 = df.query('datetime_h < "2024-11-20"')

# Define features and target
X = df2[['day_of_week', 'day', 'hour','weekend']]
y = df2['cars']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeRegressor(max_depth=11, random_state=42)
# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)

df['pred'] = model.predict(df[['day_of_week', 'day', 'hour','weekend']])
pData = df.query('datetime_h >= "2024-11-20"')
sns.lineplot(data = pData,x = "datetime_h", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_h', y='pred', label='Predicted', linestyle="--")

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

mse = mean_squared_error(pData['cars'],pData['pred'])
print(f'Mean Squared Error: {mse}')



model = DecisionTreeRegressor(random_state=42)
# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,12,13,14,15,16,17],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3,4,5,6,7,8]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, 
                           verbose=2, scoring='neg_mean_squared_error')

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

df['pred'] = best_rf.predict(df[['day_of_week', 'day', 'hour','weekend']])
pData = df.query('datetime_h >= "2024-11-20"')
sns.lineplot(data = pData,x = "datetime_h", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_h', y='pred', label='Predicted', linestyle="--")

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

mse = mean_squared_error(pData['cars'],pData['pred'])
print(f'Mean Squared Error: {mse}')
print(f'Best Parameters: {grid_search.best_params_}')



model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)

df['pred'] = model.predict(df[['day_of_week', 'day', 'hour','weekend']])
pData = df.query('datetime_h >= "2024-11-20"')
sns.lineplot(data = pData,x = "datetime_h", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_h', y='pred', label='Predicted', linestyle="--")

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

mse = mean_squared_error(pData['cars'],pData['pred'])
print(f'Mean Squared Error: {mse}')





# Get feature importances from the best model 
importances = best_rf.feature_importances_ 
feature_names = X.columns 
# Create a DataFrame for visualization 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) 
# Sort the DataFrame by importance 
importance_df = importance_df.sort_values(by='Importance', ascending=False) 
# Plot the feature importances 



##################
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3,4,5,6,7,8]
}

ts_cv = TimeSeriesSplit(
    n_splits=5
)
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=ts_cv, n_jobs=-1, 
                           verbose=2, scoring='neg_mean_squared_error')
# Fit GridSearchCV on the training data
grid_search.fit(X, y)
# Get the best model from grid search
best_rf = grid_search.best_estimator_
# Predict on the test set
y_pred = best_rf.predict(X_test)
df['pred'] = best_rf.predict(df[['day_of_week', 'day', 'hour','weekend']])
pData = df.query('datetime_h >= "2024-11-20"')
sns.lineplot(data = pData,x = "datetime_h", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_h', y='pred', label='Predicted', linestyle="--")
specific_date = '2024-11-20' 
plt.axvline(pd.to_datetime(specific_date), color='red', linestyle='--', linewidth=2, label='Event Date')

mse = mean_squared_error(pData['cars'],pData['pred'])
print(f'Mean Squared Error: {mse}')

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, 
                           verbose=2, scoring='neg_mean_squared_error')
# Fit GridSearchCV on the training data
grid_search.fit(X, y)
# Get the best model from grid search
best_rf = grid_search.best_estimator_
# Predict on the test set
y_pred = best_rf.predict(X_test)
df['pred'] = best_rf.predict(df[['day_of_week', 'day', 'month', 'hour','weekend']])
pData = df.query('datetime_f >= "2024-11-20"')
sns.lineplot(data = pData,x = "datetime_f", y = "cars", label='Actual')
sns.lineplot(data=pData, x='datetime_f', y='pred', label='Predicted', linestyle="--")



# Get feature importances from the best model 
importances = best_rf.feature_importances_ 
feature_names = X.columns 
# Create a DataFrame for visualization 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) 
# Sort the DataFrame by importance 
importance_df = importance_df.sort_values(by='Importance', ascending=False) 
# Plot the feature importances 
plt.figure(figsize=(10, 6)) 
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue') 
plt.xlabel('Importance') 
plt.ylabel('Feature') 
plt.title('Feature Importances in Random Forest Model') 
plt.gca().invert_yaxis() 
plt.show()






from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint


# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 5, 10],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, 
                                   n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', 
                                   random_state=42)

# Fit RandomizedSearchCV on the training data
random_search.fit(X, y)

# Get the best model from random search
best_rf = random_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Best Parameters: {random_search.best_params_}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Number of Cars')
plt.title('Actual vs Predicted Number of Cars')
plt.legend()
plt.show()


####################



# Load your dataset
df = pd.read_csv('/workspaces/data/msn_parking/output/commute_clean.csv')
df = df.query('name == "Level 1"')
# Ensure the datetime column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])
df['cars'] = df['total_spots'] - df['available_spots']
df['datetime_f'] = df['datetime'].dt.floor('h')
df = df.groupby('datetime_f')['cars'].max().reset_index()

# Extract features
df['day_of_week'] = df['datetime_f'].dt.day_name  # Monday=0, Sunday=6
df['day'] = df['datetime_f'].dt.day
df['month'] = df['datetime_f'].dt.month
df['hour'] = df['datetime_f'].dt.hour



# Define features and target
X = df[['day_of_week', 'day', 'month', 'hour']]
y = df['cars']




# Column names for transformation
numerical_features = []
categorical_features = ['day_of_week']

# Preprocessing for numerical data: StandardScaler
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: OneHotEncoder
categorical_transformer = OneHotEncoder()

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)], 
        remainder='passthrough'
    )

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeRegressor(max_depth = 11))
])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Extract the trained decision tree model 
clf = pipeline.named_steps['classifier'] 


# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Samples')
plt.ylabel('Number of Cars')
plt.title('Actual vs Predicted Number of Cars')
plt.legend()
plt.show()




















import pandas as pd

import seaborn as sns

import matplotlib.dates as mdates



import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('/workspaces/data/msn_parking/output/commute_clean.csv')

cleaned = (
    df
    .assign(cars = lambda x: x['total_spots'] - x['available_spots'],
         datetime = lambda x: pd.to_datetime(x['datetime']),
         date = lambda x: x['datetime'].dt.date,
         day = lambda x: x['datetime'].dt.day,
         weekday = lambda x: x['datetime'].dt.day_name(),
         weekday_n = lambda x: x['datetime'].dt.weekday)
         )
pData = (
    cleaned.groupby(['datetime','date','day','weekday','weekday_n'])['cars']
    .sum()
    .reset_index()
    .groupby(['date','day','weekday','weekday_n'])['cars'].mean()
    .reset_index()
)

pData2 = pData.groupby(['weekday','weekday_n'])['cars'].mean().reset_index()
sns.barplot(data = pData2,x = "weekday_n", y = "cars")
plt.xticks(ticks=pData2['weekday_n'], labels=pData2['weekday'])

pData2 = pData.groupby('day')['cars'].mean().reset_index().sort_values('cars')
sns.lineplot(data = pData2,x = "day", y = "cars")

# Column names for transformation
numerical_features = []
categorical_features = ['weekday']

# Preprocessing for numerical data: StandardScaler
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: OneHotEncoder
categorical_transformer = OneHotEncoder()

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)], 
        remainder='passthrough'
    )

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeRegressor(max_depth = 3))
])

# Features and target variable
X = pData[['weekday']]
y = pData['cars']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Extract the trained decision tree model 
clf = pipeline.named_steps['classifier'] 
# Retrieve feature names from the preprocessor 
encoded_feature_names = list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['weekday'])) 
feature_names = numerical_features + encoded_feature_names 
# Format the feature names to be more readable 
formatted_feature_names = [name.replace('weekday_', 'Is weekday: ') for name in feature_names] 
# Plot the decision tree with readable feature names 
plt.figure(figsize=(12, 8)) 
plot_tree(clf, feature_names=formatted_feature_names, filled=True) 
plt.show()
from sklearn import metrics
y_pred = pipeline.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.root_mean_squared_error(y_test, y_pred))




# Extract the trained decision tree model 
clf = pipeline.named_steps['classifier'] 
# Plot the decision tree 
plt.figure(figsize=(12, 8)) 
plot_tree(clf, feature_names= list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['weekday'])) + list(X.columns)[:-1], filled=True) 
plt.show()







sns.lineplot(data = pData,x = "date", y = "cars")
# Rotate date labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
# Format the x-axis to show every month
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))



from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




# Features and target variable
X = cleaned[['weekday']]
y = cleaned['cars']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

# Print the tree in text format
tree_text = export_text(clf, feature_names=list(X.columns))
print(tree_text)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=list(X.columns), class_names=['Class 0', 'Class 1'], filled=True)
plt.show()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
# Example data
df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})

# One-Hot Encoding

encoded_data = ohe.fit_transform(df[['category']])
df_encoded = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(['category']))

# Combine the original DataFrame with the encoded data
df = pd.concat([df, df_encoded], axis=1)

print(df)



import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample DataFrame with date and other features
data = {
    'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Define the preprocessing steps
def date_to_weekday(df):
    df['weekday'] = df['date'].dt.day_name()
    return df

# Apply date transformation
df = date_to_weekday(df)

# Separate features and target variable
X = df[['feature1', 'feature2', 'weekday']]
y = df['target']

# Column names for transformation
numerical_features = ['feature1', 'feature2']
categorical_features = ['weekday']

# Preprocessing for numerical data: StandardScaler
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: OneHotEncoder
categorical_transformer = OneHotEncoder()

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Print the predictions
print("Predictions:", y_pred)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame with date and other features
data = {
    'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Define the preprocessing steps
def date_to_weekday(df):
    df['weekday'] = df['date'].dt.day_name()
    return df

# Apply date transformation
df = date_to_weekday(df)

# Separate features and target variable
X = df[['feature1', 'feature2', 'weekday']]
y = df['target']

# Column names for transformation
numerical_features = ['feature1', 'feature2']
categorical_features = ['weekday']

# Preprocessing for numerical data: StandardScaler
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: OneHotEncoder
categorical_transformer = OneHotEncoder()

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline with max_depth parameter
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=3))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Extract the trained decision tree model
clf = pipeline.named_steps['classifier']

# Retrieve the scaler used in preprocessing
scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']

# Unstandardize the numerical features for plotting
feature_names = list(X.columns[:-1]) + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['weekday']))

# Create a function to reverse the standardization
def unstandardize(value, mean, scale):
    return value * scale + mean

# Unstandardize feature names (assuming numerical features are listed first)
unstandardized_feature_names = [
    f"{feature_names[i]} (orig: {unstandardize(0, scaler.mean_[i], scaler.scale_[i]):.2f} ± {scaler.scale_[i]:.2f})"
    for i in range(len(numerical_features))
] + feature_names[len(numerical_features):]

# Plot the decision tree with unstandardized feature names
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=unstandardized_feature_names, class_names=['Class 0', 'Class 1'], filled=True)
plt.show()
