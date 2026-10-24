import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = {
    "Hours": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
    "Marks": [10, 13, 20, 23, 30, 33, 40, 43, 50, 53, 60, 63, 70, 73, 80, 83, 90, 93, 100]
}

df = pd.DataFrame(data)

print("Student marks after converting into DataFrame:")
print(df)

plt.scatter(df["Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Visualization of Student Marks Data")
plt.grid(True)
plt.show()

X = df[["Hours"]]
Y = df[["Marks"]]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Model training is completed.")

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)

print("Model Evaluation Results:")
print("Actual Marks:")
print(Y_test)

print("Predicted Marks:", np.round(y_pred, 2))
print(f"Mean Absolute Error: {mae:.2f}")

try:
    hours = int(input("Enter your study hours: "))

    input_data = pd.DataFrame({
        "Hours": [hours]
    })

    pred = model.predict(input_data)
    pred = max(0, min(100, pred[0]))

    print("Predicted Marks:", np.round(pred, 2))

except ValueError:
    print("Invalid input. Please enter a valid number ")
