import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/student_data.csv")

# ---------------------------------------
# Graph 1: Study Hours vs Final Marks
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["StudyHours"],
    data["FinalMarks"]
)

plt.xlabel("Study Hours per Day")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")

plt.grid(True)

plt.show()


# ---------------------------------------
# Graph 2: Attendance vs Final Marks
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Attendance"],
    data["FinalMarks"]
)

plt.xlabel("Attendance Percentage")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks")

plt.grid(True)

plt.show()