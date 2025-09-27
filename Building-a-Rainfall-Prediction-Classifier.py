import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("weatherAUS_2.csv")
print(df.head())
print(df.count())

# Drop all rows with missing values
df = df.dropna()
print(df.info())
print(df.columns)

# Data leakage considerations
df = df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})

# Location selection
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia'])]
df.info()

# Extracting a seasonality feature
def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

# Map the dates to seasons and drop the Date column
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df = df.drop(columns=['Date'])
print(df.head())

# Define feature and target
X = df.drop(columns='Season', axis=1)
y = df['Season']

# Check class balance
print(y.value_counts())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Detect numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Pipeline with Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    scoring='accuracy',
    verbose=2
)

# Fit Random Forest
grid_search.fit(X_train, y_train)
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Test score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# Predictions
y_pred = grid_search.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Feature importances
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_
feature_names = numeric_features + list(
    grid_search.best_estimator_['preprocessor']
        .named_transformers_['cat']
        .get_feature_names_out(categorical_features)
)

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})\
                .sort_values(by='Importance', ascending=False)

N = 20
top_features = importance_df.head(N)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(f'Top {N} Most Important Features in predicting Season')
plt.xlabel('Importance Score')
plt.show()

# Switch to Logistic Regression
pipeline.set_params(classifier=LogisticRegression(random_state=42))
grid_search.estimator = pipeline
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}
grid_search.param_grid = param_grid

# Fit Logistic Regression
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

# Classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()