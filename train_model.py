import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

data = pd.read_csv("dataset/student_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# ---------------------------------------
# 2. Select Input and Output
# ---------------------------------------

X = data[
    [
        "Attendance",
        "StudyHours",
        "PreviousMarks",
        "AssignmentScore",
        "InternalMarks"
    ]
]

y = data["FinalMarks"]

# ---------------------------------------
# 3. Split Dataset
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------
# 4. Create Machine Learning Model
# ---------------------------------------

model = LinearRegression()

# ---------------------------------------
# 5. Train Model
# ---------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ---------------------------------------
# 6. Test Model
# ---------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------
# 7. Evaluate Model
# ---------------------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# ---------------------------------------
# 8. Save Model
# ---------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/student_model.pkl")

print("\nModel saved successfully!")
print("Location: model/student_model.pkl")