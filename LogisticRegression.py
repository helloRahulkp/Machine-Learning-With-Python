import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#loading data 
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

# print(churn_df.sample(5))
# print(churn_df.columns)
# print(churn_df.describe())

# create a sub set of data with fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip' and of course 'churn'.

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
# print(churn_df.sample(5))
# print(churn_df.describe())

churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
# print(X.shape)
# print(X[0:5])
y = np.asarray(churn_df['churn'])
# print(y.shape)
# print(y[0:5])

#normalize the dataset in order to have all the features at the same scale

X_norm = StandardScaler().fit(X).transform(X)
# print(X_norm[0:5])

# separate a part of the data for testing and the remaining for training.

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

# Let's build the model using LogisticRegression from the Scikit-learn package and fit our model with train data set.

LR = LogisticRegression().fit(X_train, y_train)

# Let us predict the churn parameter for the test data set.
yhat = LR.predict(X_test)
# print(yhat[0:5])

# To understand this prediction, we can also have a look at the prediction probability of data point of the test data set. Use the function predict_proba
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]


coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Log loss (Logarithmic loss), also known as Binary Cross entropy loss
log_loss(y_test, yhat_prob)
print("Log Loss: ", log_loss(y_test, yhat_prob))


#HomeWork
# Try to attempt the following questions yourself based on what you learnt in this lab.

# a. Let us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case?
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.6039104035600186.

# b. Let us assume we add the feature 'wireless' to the original set of input features. What will the value of log loss be in this case?
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7227054293985518.

# c. What happens to the log loss value if we add both "callcard" and "wireless" to the input features?
# Hint
# Reuse all the code statements above after modifying the value of churn_df. Make sure to edit the list of features feeding the variable X. The expected answer is 0.7760557225417114

# d. What happens to the log loss if we remove the feature 'equip' from the original set of input features?
# Hint
# Reuse all the code statements above after modifying the value of churn_df Make sure to edit the list of features feeding the variable X. The expected answer is 0.5302427350245369

# e. What happens to the log loss if we remove the features 'income' and 'employ' from the original set of input features?
# Hint
