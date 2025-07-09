import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

# The data set being used for this lab is the "Obesity Risk Prediction" data set publically 
# available on UCI Library under the CCA 4.0 license. The data set has 17 attributes in total 
# along with 2,111 samples.

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
print("Sample data:", data.sample(5))
print("Data columns:", data.columns)

# EDA 
# Visualize the distribution of the target variable to understand the class balance.

sns.countplot(y='NObeyesdad', data = data)
plt.title('Distribution of Obesity Levels')
plt.show()

# Check for null values, and display a summary of the dataset 
Null_values = data.isnull().sum()
print("Null values in each column:\n", Null_values)
# print(data.isnull())
print("Data info:\n", data.info())
data_description = data.describe()
print("Data description:\n", data_description)

# Preprocessing the data
# Feature scaling
# Scale the numerical features to standardize their ranges for better model performance.

continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
print("Continuous columns:", continuous_columns)

scalar = StandardScaler()
scaled_features = scalar.fit_transform(data[continuous_columns])

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scalar.get_feature_names_out(continuous_columns))
print("Scaled features:\n", scaled_df.sample(5))

# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns),scaled_df], axis=1)

# Convert categorical variables into numerical format using one-hot encoding.

categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:\n", categorical_columns)
categorical_columns.remove('NObeyesdad')  # Exclude the target variable from one-hot encoding
print("Categorical columns after excluding target variable:\n", categorical_columns)

# Applying one-hot encoding
encoder  = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Convert the encoded features back to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
print("Encoded features:\n", encoded_df.sample(5))

# combining with orginal dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
print("Prepared data:\n", prepped_data.sample(5))

# Encode the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
print(prepped_data.head())

# Splitting the data into training and testing sets
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Split the data into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) 
# and observe the impact on model performance.

for test_size in [0.1, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model_ova.fit(X_train, y_train)
    y_pred = model_ova.predict(X_test)
    print(f"Test Size: {test_size}")
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression 
# model. Also try for the One vs One model.

# Feature importance
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()


# For One vs One model
# Collect all coefficients from each underlying binary classifier
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])

# Now take the mean across all those classifiers
feature_importance = np.mean(np.abs(coefs), axis=0)

# Plot feature importance
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance (One-vs-One)")
plt.xlabel("Importance")
plt.show()

# Write a function obesity_risk_pipeline to automate the entire pipeline:

#     1.Loading and preprocessing the data
#     2.Training the model
#     3.Evaluating the model

# The function should accept the file path and test set size as the input arguments.

def obesity_risk_pipeline(data_path, test_size=0.2):
    # Load data
    data = pd.read_csv(data_path)

    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    
    # Converting to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    
    # Combining with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column
    
    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    
    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Preparing final dataset
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']
    print("Prepared data shape:", X.shape, "Target shape:", y.shape)
    print("Prepared data sample:\n", prepped_data.sample(5))
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Training and evaluation
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Call the pipeline function with file_path
obesity_risk_pipeline(file_path, test_size=0.2)
