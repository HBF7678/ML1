import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the dataset
X = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]).reshape(-1, 1)
y = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the speed for a 23-year-old car
age_of_car = np.array([23]).reshape(-1, 1)
predicted_speed = model.predict(age_of_car)

# Visualization
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
plt.scatter(age_of_car, predicted_speed, color='green', label='23-year-old car prediction')
plt.xlabel('Age of Car')
plt.ylabel('Speed')
plt.title('Linear Regression: Age of Car vs Speed')
plt.legend()
plt.show()

# Output the predicted speed
predicted_speed
