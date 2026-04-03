# -------------------------------------------------------
# Student Score Prediction using Linear Regression
# -------------------------------------------------------

# step 1: Import all important libraries

# Step 2: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# Step 3: Load Dataset

df = pd.read_csv("Student_performance_data _.csv")

print("Dataset Shape:", df.shape)       # rows and columns
print("\nFirst 5 rows:\n", df.head())
print("\nColumn Names:\n", df.columns.tolist())


# Step 4: Select Features and Target
# We predict GPA based on study hours, absences, tutoring, etc.

features = ['Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport']
target   = 'GPA'

X = df[features]
y = df[target]

print("\nFeatures used:", features)
print("Target       :", target)


# Step 5: Split into Train and Test sets (80% train, 20% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining size:", X_train.shape[0])
print("Testing size :", X_test.shape[0])


# Step 6: Train the Model

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Step 7: Make Predictions and Evaluate

y_pred = model.predict(X_test)

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nR² Score (Accuracy) : {r2:.4f}  →  {r2*100:.1f}%")
print(f"Mean Absolute Error : {mae:.4f}")


# Step 8: Predict for a New Student

new_student = pd.DataFrame({
    'Age'            : [17],
    'StudyTimeWeekly': [15],
    'Absences'       : [3],
    'Tutoring'       : [1],
    'ParentalSupport': [2]
})

predicted_gpa = model.predict(new_student)[0]
print(f"\nPredicted GPA for new student: {predicted_gpa:.2f}")

# Step 9: Plot - Actual vs Predicted GPA

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA - Nikhil Sabat")
plt.legend()
plt.tight_layout()
plt.savefig("student_gpa_prediction.png", dpi=150)
plt.show()

print("\nDone! Graph saved as student_gpa_prediction.png")
