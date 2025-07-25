"""
Code demonstrating the basics of linear regression using scikit-learn.

This code first imports the necessary libraries, then loads the data from a CSV file.
The data is then split into training and test sets after adding some noise, and a LinearRegression object is
instantiate and fit to the training data. The model is then evaluated using the
mean_squared_error, r2_score, and mean_absolute_error functions from scikit-learn.
Finally, the model is used to make predictions on the test data, and the results are
plotted using matplotlib.
"""
#Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the data
data = pd.read_csv("ML Foundations/linear_regression_dataset.csv")
X = data.drop("target", axis=1)
X += np.random.normal(0, 0.1, size=X.shape)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Evaluating the model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

#Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()