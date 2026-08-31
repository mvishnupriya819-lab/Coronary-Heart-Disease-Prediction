import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================
# 1. LOAD DATASET
# =========================================

df = pd.read_csv("framingham.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# =========================================
# 2. SELECT INPUT FEATURES
# =========================================

features = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose"
]

target = "TenYearCHD"


X = df[features]
y = df[target]


# =========================================
# 3. FILL MISSING VALUES
# =========================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features
)

print("Missing values handled successfully!")


# =========================================
# 4. SPLIT DATA
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================
# 5. SCALE THE DATA
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Data scaling completed!")


# =========================================
# 6. CREATE MODEL
# =========================================

model = LogisticRegression(
    max_iter=1000
)


# =========================================
# 7. TRAIN MODEL
# =========================================

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed!")


# =========================================
# 8. TEST MODEL
# =========================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================
# 9. SAVE MODEL
# =========================================

with open("lg_CHD_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as:")
print("lg_CHD_model.pkl")


# =========================================
# 10. SAVE SCALER
# =========================================

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Scaler saved as:")
print("scaler.pkl")


print("\nTRAINING COMPLETED SUCCESSFULLY!")