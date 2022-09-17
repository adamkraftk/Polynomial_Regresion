import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read data from file
file = pd.read_csv("data.csv")
X = file[["Salary"]]
y = file[["Years of experience"]]

x_train = X[:-6]
y_train = y[:-6]

x_test = X[-4:]
y_test = y[-4:]

# Set PolynomialFeatures to degree 2 and store in the variable pre_proces
pre_process = PolynomialFeatures(degree=2)

# Transform our x input to 1, x and x^2
X_poly = pre_process.fit_transform(y_train)

# Use the function to get a line for prediction
pr_model = LinearRegression()

# Fit our preprocessed data to the polynomial regression model
pr_model.fit(X_poly, y_train)

# Help train the y values
y_pred = pr_model.predict(X_poly)

# Use this function to make predictions
y_new = pr_model.predict(pre_process.fit_transform(y_test))

# Plot our model on our data
plt.scatter(x_train, y_train, c = "g")
plt.scatter(x_test, y_new, c = "red")
plt.xlabel("Salaries")
plt.ylabel("Years of experience")
plt.plot(x_train, y_pred)
plt.show()


