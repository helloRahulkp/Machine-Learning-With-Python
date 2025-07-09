import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)
print(df.sample(5))
print(df.shape)
print(df.columns)

print("Summary statistics:\n", df.describe())
print(df.tail(5))


print("Select a few features that might be indicative of CO2 emission to explore more")
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.sample(5))

print("Consider the histograms for each of these features")
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()


print("Plotting the scatter plots to see the relationship between FUELCONSUMPTION_COMB and CO2EMISSIONS")

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

print("Plotting the scatter plots from zero to 27 for ENGINESIZE and CO2EMISSIONS")
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)
plt.show()

print("Plot CYLINDER against CO2 Emission, to see how linear their relationship is:")
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()

print("Extract the input feature and labels from the dataset")
x = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

print("Create train and test datasets")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("The outputs are one-dimensional NumPy arrays or vectors")

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
print("Type of X_train:", type(X_train))
print("Type of y_train:", type(y_train))

print("Build a simple linear regression model")

from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

print("Visualize model output")
# The regression model is the line given by y = intercept + coefficient * x.
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.intercept_ + regressor.coef_[0] * X_train, '-r', label='Regression Line')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


print("Model evaluation")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))


print("Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.")
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r', label='Regression Line')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

print("Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets")
x = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("Use the model to make test predictions on the fuel consumption testing data")
y_test_ = regressor.predict(X_test.reshape(-1,1))

print("Calculate and print the Mean Squared Error of the test predictions")
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))