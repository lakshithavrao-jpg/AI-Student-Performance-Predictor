import streamlit as st
import joblib
import pandas as pd


# ---------------------------------------
# Load Model
# ---------------------------------------

model = joblib.load("model/student_model.pkl")


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("🎓 AI Student Performance Predictor")

st.write(
    "Enter the student's academic details below "
    "to predict their final examination marks."
)


st.divider()


# ---------------------------------------
# Input Section
# ---------------------------------------

st.subheader("Student Details")


attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)


study_hours = st.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)


previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)


assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)


internal_marks = st.number_input(
    "Internal Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)


st.divider()


# ---------------------------------------
# Prediction Button
# ---------------------------------------

if st.button("🔮 Predict Performance"):

    input_data = pd.DataFrame(
        {
            "Attendance": [attendance],
            "StudyHours": [study_hours],
            "PreviousMarks": [previous_marks],
            "AssignmentScore": [assignment_score],
            "InternalMarks": [internal_marks]
        }
    )

    # Make prediction
    prediction = model.predict(input_data)

    final_marks = prediction[0]

    # Keep marks within 0-100
    final_marks = max(0, min(100, final_marks))


    # -----------------------------------
    # Performance Category
    # -----------------------------------

    if final_marks >= 90:
        performance = "Excellent 🌟"
    elif final_marks >= 75:
        performance = "Good 👍"
    elif final_marks >= 60:
        performance = "Average 📚"
    else:
        performance = "Needs Improvement ⚠️"


    # -----------------------------------
    # Display Result
    # -----------------------------------

    st.success("Prediction completed!")

    st.metric(
        "Predicted Final Marks",
        f"{final_marks:.2f}/100"
    )

    st.subheader("Performance Category")

    st.info(performance)


    # -----------------------------------
    # Suggestions
    # -----------------------------------

    st.subheader("Suggestions")

    if attendance < 75:
        st.warning(
            "Attendance is low. Try to maintain attendance above 75%."
        )

    if study_hours < 4:
        st.warning(
            "Consider increasing daily study hours."
        )

    if assignment_score < 60:
        st.warning(
            "Try to improve assignment performance."
        )

    if internal_marks < 60:
        st.warning(
            "More preparation may be needed for internal examinations."
        )

    if (
        attendance >= 75
        and study_hours >= 4
        and assignment_score >= 60
        and internal_marks >= 60
    ):
        st.success(
            "The student's academic indicators are currently good."
        )