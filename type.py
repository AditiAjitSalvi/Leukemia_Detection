# svm_leukemia_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load Dataset
df = pd.read_csv("synthetic_bonemarrow_leukemia_dataset.csv")

print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)
print(df.head())

# 2️⃣ Split Features & Labels
X = df.drop("Type", axis=1)
y = df["Type"]

# Encode Target Labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4️⃣ Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ SVM Model + Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Best Parameters:", grid.best_params_)

best_svm = grid.best_estimator_

# 6️⃣ Evaluate Model
y_pred = best_svm.predict(X_test_scaled)
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

# 7️⃣ Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix - Leukemia Type Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8️⃣ Save Model + Scaler + Encoder
joblib.dump(best_svm, "svm_leukemia_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\n✅ Model, Scaler, and Encoder saved successfully!")

# 9️⃣ Function for Manual Input Prediction
def predict_leukemia_type():
    print("\n🔹 Enter Bone Marrow / CBC values to predict Leukemia Type 🔹")
    wbc = float(input("WBC count (cells/µL): "))
    rbc = float(input("RBC count (million/µL): "))
    hb = float(input("Hemoglobin (g/dL): "))
    platelet = float(input("Platelet count (×10³/µL): "))
    myeloblast = float(input("Myeloblast %: "))
    lymphoblast = float(input("Lymphoblast %: "))
    me_ratio = float(input("M:E Ratio: "))
    ldh = float(input("LDH (U/L): "))
    age = float(input("Age (years): "))

    # Prepare input array
    features = np.array([[wbc, rbc, hb, platelet, myeloblast, lymphoblast, me_ratio, ldh, age]])

    # Load saved model, scaler, encoder
    model = joblib.load("svm_leukemia_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    # Scale input
    features_scaled = scaler.transform(features)
    # Prepare input array
    """features = np.array([[wbc, rbc, hb, platelet, myeloblast, lymphoblast, me_ratio, ldh, age]])

    # ✅ Convert to DataFrame with column names (must match training data)
    columns = X.columns.tolist()  # ['WBC','RBC','Hb','Platelet','Myeloblast', 'Lymphoblast', 'M:E Ratio', 'LDH', 'Age']
    features_df = pd.DataFrame(features, columns=columns)

    # Scale input
    features_scaled = scaler.transform(features_df)"""


    # Predict
    pred = model.predict(features_scaled)
    leukemia_type = encoder.inverse_transform(pred)[0]

    if leukemia_type == "AML":
      print("\n🩸 Predicted Leukemia Type → Acute Myeloid Leukemia")
    elif leukemia_type == "ALL":
        print("\n🩸 Predicted Leukemia Type → Acute Lymphoblastic Leukemia")
    elif leukemia_type == "CML":
        print("\n🩸 Predicted Leukemia Type → Chronic Myeloid Leukemia")
    elif leukemia_type == "CLL":
        print("\n🩸 Predicted Leukemia Type → Chronic Lymphocytic Leukemia")
    elif leukemia_type == "Normal":
        print("\n🩸 Predicted Leukemia Type → Normal Bone Marrow / Healthy Sample")
    else:
        print("Unknown leukemia type")
        
    print("\n🩸 Predicted Leukemia Type →", leukemia_type)
    print("------------------------------------------------")

# 🔟 Run interactive prediction
predict_leukemia_type()
