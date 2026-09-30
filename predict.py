import joblib

# Load trained model
model = joblib.load("model/student_model.pkl")

print("Student Performance Predictor")
print("--------------------------------")

# Get student details
attendance = float(input("Enter attendance percentage: "))
study_hours = float(input("Enter study hours per day: "))
previous_marks = float(input("Enter previous marks: "))
assignment_score = float(input("Enter assignment score: "))
internal_marks = float(input("Enter internal marks: "))

# Prepare input
student_data = [[
    attendance,
    study_hours,
    previous_marks,
    assignment_score,
    internal_marks
]]

# Predict final marks
prediction = model.predict(student_data)

final_marks = prediction[0]

# Performance category
if final_marks >= 90:
    performance = "Excellent"
elif final_marks >= 75:
    performance = "Good"
elif final_marks >= 60:
    performance = "Average"
else:
    performance = "Needs Improvement"

print("\n--------------------------------")
print("Predicted Final Marks:", round(final_marks, 2))
print("Performance:", performance)
print("--------------------------------")