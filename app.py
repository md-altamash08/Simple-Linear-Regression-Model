# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Title for the app
st.title("Linear Regression Demo")

# Step 1: Create a small dataset
st.write("### Dataset")
data = {
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Target': [3, 4, 2, 5, 6, 7, 8, 9, 10, 12]
}
df = pd.DataFrame(data)
st.write(df)

# Step 2: Split the data into features and target
X = df[['Feature']]  # Feature column
y = df['Target']     # Target column

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display performance metrics
st.write("### Model Performance Metrics")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Step 7: Visualize the regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression: Feature vs Target')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
st.pyplot(plt)
