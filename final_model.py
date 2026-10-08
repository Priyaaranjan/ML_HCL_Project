import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "dataset/agaricus-lepiota.data"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# COLUMN NAMES
# ============================================================

columns = [
    "class",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    DATASET_PATH,
    header=None,
    names=columns
)

print("Original dataset shape:", df.shape)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

df = df.replace("?", float("nan"))

missing_count = df.isna().sum().sum()

print("Missing values found:", missing_count)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)

df = df.drop_duplicates()

print("Dataset shape after duplicate removal:", df.shape)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

print("\nChecking constant features...")

X = df.drop(columns=["class"])
y = df["class"]

constant_features = [
    column for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

print("Constant features:", constant_features)

if constant_features:
    X = X.drop(columns=constant_features)

print("Features after removing constant columns:", X.shape[1])


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.columns.tolist()

print("\nCategorical features:", len(categorical_features))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_features
        )
    ]
)


# ============================================================
# FEATURE SELECTION + RANDOM FOREST
# ============================================================

final_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "feature_selection",
        SelectKBest(
            score_func=chi2,
            k=20
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

final_pipeline.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = final_pipeline.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    pos_label="p"
)
recall = recall_score(
    y_test,
    y_pred,
    pos_label="p"
)
f1 = f1_score(
    y_test,
    y_pred,
    pos_label="p"
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["e", "p"]
)


print("\n" + "=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1 Score  : {f1:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Edible", "Poisonous"]
    )
)


# ============================================================
# GET SELECTED FEATURES
# ============================================================

encoder = (
    final_pipeline
    .named_steps["preprocessing"]
    .named_transformers_["categorical"]
    .named_steps["encoder"]
)

encoded_feature_names = encoder.get_feature_names_out()

selector = final_pipeline.named_steps["feature_selection"]

selected_mask = selector.get_support()

selected_features = encoded_feature_names[selected_mask]

print("\n" + "=" * 60)
print("SELECTED FEATURES")
print("=" * 60)

print("Total encoded features:", len(encoded_feature_names))
print("Selected features:", len(selected_features))

for feature in selected_features:
    print(feature)


# ============================================================
# SAVE FINAL PIPELINE
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "final_mushroom_pipeline.pkl"
)

joblib.dump(
    final_pipeline,
    model_path
)

print("\nFinal pipeline saved to:")
print(model_path)


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

selected_features_path = os.path.join(
    MODEL_DIR,
    "selected_features.txt"
)

with open(
    selected_features_path,
    "w",
    encoding="utf-8"
) as file:

    for feature in selected_features:
        file.write(feature + "\n")

print("Selected features saved to:")
print(selected_features_path)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

metrics = {
    "model": "Random Forest",
    "feature_selection": "Chi-Square SelectKBest",
    "selected_features": 20,
    "total_encoded_features": int(len(encoded_feature_names)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy),
    "precision_poisonous": float(precision),
    "recall_poisonous": float(recall),
    "f1_poisonous": float(f1),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    MODEL_DIR,
    "final_model_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("Model metrics saved to:")
print(metrics_path)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL PIPELINE READY")
print("=" * 60)
print("Random Forest + Chi-Square Feature Selection")
print("20 selected features from encoded features")
print("80% training / 20% testing")
print("Preprocessing is included inside the saved pipeline.")