import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

loc = '/media/rahulkp/HDD/Learning/IBM AI Engineering/Lab/teleCust1000t.csv'
df = pd.read_csv(loc)
print(df.head())
print("Sample: \n",df.sample(5))

# the class-wise distribution of the data set.
print("Class-wise distribution of the data set: \n", df['custcat'].value_counts())
# Visualizing the class-wise distribution
sns.countplot(x='custcat', data=df)
plt.title('Class-wise Distribution of the Data Set')
plt.xlabel('Customer Category')
plt.ylabel('Count')
plt.show()

# correlation map of the data set to determine how the different features are related to each other.
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Map of the Data Set')
plt.show()

# descending order of their absolute correlation values with respect to the target field.

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print("Features sorted by their absolute correlation with the target field: \n", correlation_values)

# separate the data into the input data set and the target data set.
X = df.drop('custcat', axis=1)
y = df['custcat']

# Normalize Data
X_norm = StandardScaler().fit_transform(X)

# separate the training and the testing data. 
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# KNN Classification
# Initially, you may start by using a small value as the value of k, say k = 4.
k = 3
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# Predicting
yhat = knn_model.predict(X_test)

# accuracy classification score is a function that computes subset accuracy
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# Choosing the correct value of k
# Check the performance of the model for 10 values of k, ranging from 1-9.

Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# Plot the model accuracy for a different number of neighbors

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 

# Plot the variation of the accuracy score for the training set for 100 value of Ks.
Ks =100
acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_train)
    acc[n-1] = accuracy_score(y_train, yhat)
    std_acc[n-1] = np.std(yhat==y_train)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),acc,'g')
plt.fill_between(range(1,Ks),acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()