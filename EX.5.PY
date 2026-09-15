# ============================================================
# EXPERIMENT: URL Classification using Support Vector Machine
# Dataset: phishing_url_dataset-selected-columns.csv
# ============================================================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv("D:\\SEM 3\\phishing_url_dataset-selected-columns.csv")


print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 3. Display Dataset Information
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())


# ============================================================
# 4. Check Missing Values
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# Remove rows containing missing values
df = df.dropna()


# ============================================================
# 5. Separate Features and Target
# ============================================================

# valid_url is the target column
# 0 = Malicious
# 1 = Legitimate

X = df.drop("valid_url", axis=1)
y = df["valid_url"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("valid_url")


# ============================================================
# 6. Display Class Distribution
# ============================================================

print("\nClass Distribution:")
print(y.value_counts())


# ============================================================
# 7. Split Dataset into Training and Testing
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ============================================================
# 8. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. Create SVM Model
# ============================================================

model = SVC(
    kernel="linear",
    C=1.0
)


# ============================================================
# 10. Train SVM Model
# ============================================================

print("\nTraining SVM Model...")

model.fit(
    X_train_scaled,
    y_train
)

print("SVM Model Training Completed!")


# ============================================================
# 11. Make Predictions
# ============================================================

y_pred = model.predict(X_test_scaled)


# ============================================================
# 12. Calculate Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")


# ============================================================
# 13. Calculate Precision
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nPrecision:")
print(round(precision * 100, 2), "%")


# ============================================================
# 14. Calculate Recall
# ============================================================

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nRecall:")
print(round(recall * 100, 2), "%")


# ============================================================
# 15. Calculate F1 Score
# ============================================================

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nF1 Score:")
print(round(f1 * 100, 2), "%")


# ============================================================
# 16. Classification Report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Malicious",
            "Legitimate"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. Display Confusion Matrix
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Malicious",
        "Legitimate"
    ]
)

disp.plot()

plt.title("SVM - URL Classification")
plt.show()


# ============================================================
# 19. Actual vs Predicted Graph
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values,
    label="Actual"
)

plt.plot(
    y_pred,
    label="Predicted"
)

plt.title("Actual vs Predicted URL Classification")
plt.xlabel("Test Samples")
plt.ylabel("Class")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 20. Test a New URL
# ============================================================
# Since the dataset does not contain the original URL text,
# we use its numerical features for prediction.

# Example feature values:
# url_length
# at_symbol
# sensitive_words_count
# path_length
# isHttps
# nb_dots
# nb_hyphens
# nb_and
# nb_or

new_url = [[
    50,     # url_length
    0,      # at_symbol
    1,      # sensitive_words_count
    20,     # path_length
    1,      # isHttps
    2,      # nb_dots
    1,      # nb_hyphens
    0,      # nb_and
    0       # nb_or
]]

new_url_scaled = scaler.transform(new_url)

prediction = model.predict(new_url_scaled)


if prediction[0] == 0:
    print("\nNew URL Prediction: Malicious")
else:
    print("\nNew URL Prediction: Legitimate")


# ============================================================
# 21. Final Result
# ============================================================

print("\n========================================")
print("          FINAL RESULT")
print("========================================")

print("Algorithm : Support Vector Machine")
print("Kernel    : Linear")
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")

print("========================================")