import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# Load the dataset
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
print(my_data.head(5))
print(my_data.sample(5))
print(my_data.columns)
# my_data.set_index('Age', inplace=True)
# print(my_data.sample(5))

# Data Analysis and pre-processing
# You should apply some basic analytics steps to understand the data better. 
# First, let us gather some basic information about the dataset. 

print("data info:\n",my_data.info())
print("data describe:\n",my_data.describe())
print("data isnull:\n",my_data.isnull().sum())
print("data value_counts in drug:\n",my_data['Drug'].value_counts())

# Data info This tells us that 4 out of the 6 features of this dataset are 
# categorical, which will have to be converted into numerical ones to be used for modeling. 
# For this, we can make use of LabelEncoder from the Scikit-Learn library.

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
print("data after label encoding:\n", my_data.sample(5))
print("check is there any null values:\n", my_data.isnull().sum())

# To evaluate the correlation of the target variable with the input features, 
# it will be convenient to map the different drugs to a numerical value. 
# Execute the following cell to achieve the same.

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
print("data after mapping drug to numerical value:\n", my_data.sample(5))

# You can now use the corr() function to find the correlation of the input variables with 
# the target variable.

# Correlation matrix
correlation_matrix = my_data.corr(numeric_only=True)
print("\nCorrelation matrix:\n", correlation_matrix)
# Print correlation of all features with the target variable
print("\nCorrelation of features with Drug_num:\n", correlation_matrix['Drug_num'].sort_values(ascending=False))

# Visualize the full correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Write the code to find the correlation of the input variables with the 
# target variable and identify the features most significantly affecting the target.
my_data.drop('Drug',axis=1).corr()['Drug_num']
# Data Visualization
# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Drug', data=my_data, palette='viridis')
plt.title('Distribution of Drug Types')
plt.xlabel('Drug Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# the distribution of the dataset by plotting the count of the records with each drug recommendation. 
category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()

# modeling this dataset with a Decision tree classifier, 
# we first split the dataset into training and testing subsets. 
# For this, we separate the target variable from the input variables.
X = my_data.drop(['Drug', 'Drug_num'], axis=1)
y = my_data['Drug_num']

# Separate the training data from the testing data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# define the Decision tree classifier as drugTree and train it with the training data.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

# Now that we have trained the decision tree, we can use it to generate the predictions on the test set.
tree_predictions = drugTree.predict(X_testset)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))
# Visualize the decision tree
plt.figure(figsize=(12, 8))
plt.title("Decision Tree for Drug Recommendation")
plot_tree(drugTree)
plt.show()

# If the max depth of the tree is reduced to 3, how would the performance of the model be affected?
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
drugTree.fit(X_trainset,y_trainset)
tree_predictions = drugTree.predict(X_testset)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

# Along similar lines, identify the decision criteria for all other classes.
# Feature set and target
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = my_data['Drug']  # Using original labels for interpretation

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Train Decision Tree model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_train, y_train)

# Predict and evaluate
predTree = drugTree.predict(X_test)
print("\nAccuracy:", metrics.accuracy_score(y_test, predTree))

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(drugTree,
          feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
          class_names=drugTree.classes_,
          filled=True,
          rounded=True)
plt.title("Decision Tree for Drug Classification")
plt.show()
