import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_selection import SelectKBest, chi2


# ============================================================
# 1. COLUMN NAMES
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
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "dataset/agaricus-lepiota.data",
    header=None,
    names=columns
)

print("=" * 70)
print("MUSHROOM EDIBILITY - MODEL COMPARISON")
print("=" * 70)

print("\nOriginal dataset shape:")
print(data.shape)


# ============================================================
# 3. HANDLE UNKNOWN VALUES
# ============================================================

data = data.replace("?", np.nan)

print("\nMissing values after converting '?' to NaN:")

print(data.isnull().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(data)

data = data.drop_duplicates()

after_duplicates = len(data)

print("\nDuplicate records removed:")
print(before_duplicates - after_duplicates)

print("Dataset shape after duplicate removal:")
print(data.shape)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("class", axis=1)
y = data["class"]


# ============================================================
# 6. REMOVE CONSTANT FEATURE
# ============================================================

constant_features = []

for column in X.columns:

    if X[column].nunique(dropna=True) <= 1:
        constant_features.append(column)

print("\nConstant features:")
print(constant_features)

X = X.drop(columns=constant_features)


# ============================================================
# 7. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.columns.tolist()

print("\nNumber of categorical features:")
print(len(categorical_features))


# ============================================================
# 8. TRAIN / TEST SPLIT
# 80% TRAIN
# 20% TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(len(X_train))

print("Testing samples:")
print(len(X_test))


# ============================================================
# 9. PREPROCESSING
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
# 10. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = pipeline

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Edible",
                "Poisonous"
            ]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 12. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(best_model_name)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "model/best_mushroom_model.pkl"
)

print("\nBest model saved as:")
print("model/best_mushroom_model.pkl")


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model/model_comparison_results.csv",
    index=False
)

print("\nModel comparison results saved as:")
print("model/model_comparison_results.csv")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Original records:", before_duplicates)
print("Final records:", len(data))
print("Removed duplicates:", before_duplicates - len(data))

print("\nConstant features removed:")
print(constant_features)

print("\nBest model:")
print(best_model_name)

print(
    "\nThe model is optimized for identifying poisonous mushrooms "
    "because false negatives can create a food-safety risk."
)

print("\nModel comparison completed successfully.")