import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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
# 2. LOAD DATA
# ============================================================

data = pd.read_csv(
    "dataset/agaricus-lepiota.data",
    header=None,
    names=columns
)

print("=" * 70)
print("FEATURE SELECTION AND PCA ANALYSIS")
print("=" * 70)


# ============================================================
# 3. HANDLE UNKNOWN VALUES
# ============================================================

data = data.replace("?", np.nan)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

data = data.drop_duplicates()


# ============================================================
# 5. REMOVE CONSTANT FEATURES
# ============================================================

X = data.drop("class", axis=1)
y = data["class"]

constant_features = [
    column
    for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_features)

print("\nRemoved constant features:")
print(constant_features)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

categorical_features = X.columns.tolist()

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
# 8. BASELINE RANDOM FOREST
# ============================================================

baseline_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
])

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_pred,
    pos_label="p"
)

baseline_recall = recall_score(
    y_test,
    baseline_pred,
    pos_label="p"
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred,
    pos_label="p"
)

print("\n" + "=" * 70)
print("BASELINE RANDOM FOREST")
print("=" * 70)

print("Accuracy :", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall   :", baseline_recall)
print("F1 Score :", baseline_f1)


# ============================================================
# 9. TRANSFORM DATA FOR FEATURE SELECTION
# ============================================================

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\nEncoded feature count:")
print(X_train_encoded.shape[1])


# ============================================================
# 10. CHI-SQUARE FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("CHI-SQUARE FEATURE SELECTION")
print("=" * 70)

feature_counts = [
    10,
    20,
    30,
    40,
    50
]

feature_results = []

for k in feature_counts:

    if k >= X_train_encoded.shape[1]:
        continue

    selector = SelectKBest(
        score_func=chi2,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train_encoded,
        y_train
    )

    X_test_selected = selector.transform(
        X_test_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="p"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="p"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="p"
    )

    feature_results.append({
        "Features": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    print(
        f"\nTop {k} features:"
    )

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1 Score : {f1:.6f}"
    )


# ============================================================
# 11. PCA
# ============================================================

print("\n" + "=" * 70)
print("PCA ANALYSIS")
print("=" * 70)

pca_components = [
    10,
    20,
    30,
    40
]

pca_results = []

for components in pca_components:

    if components >= X_train_encoded.shape[1]:
        continue

    pca = PCA(
        n_components=components,
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train_encoded.toarray()
        if hasattr(X_train_encoded, "toarray")
        else X_train_encoded
    )

    X_test_pca = pca.transform(
        X_test_encoded.toarray()
        if hasattr(X_test_encoded, "toarray")
        else X_test_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_pca,
        y_train
    )

    predictions = model.predict(
        X_test_pca
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="p"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="p"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="p"
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    pca_results.append({
        "Components": components,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Explained Variance": explained_variance
    })

    print(
        f"\nPCA Components: {components}"
    )

    print(
        f"Accuracy          : {accuracy:.6f}"
    )

    print(
        f"Precision         : {precision:.6f}"
    )

    print(
        f"Recall            : {recall:.6f}"
    )

    print(
        f"F1 Score          : {f1:.6f}"
    )

    print(
        f"Explained Variance: {explained_variance:.6f}"
    )


# ============================================================
# 12. RESULTS
# ============================================================

feature_results_df = pd.DataFrame(
    feature_results
)

pca_results_df = pd.DataFrame(
    pca_results
)


print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    feature_results_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print(
    pca_results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

feature_results_df.to_csv(
    "model/feature_selection_results.csv",
    index=False
)

pca_results_df.to_csv(
    "model/pca_results.csv",
    index=False
)


# ============================================================
# 14. CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    """
The original dataset contains categorical features.
One-hot encoding was used to convert them into numerical
features.

Chi-square feature selection was tested to determine whether
a smaller subset of encoded features could maintain or improve
classification performance.

PCA was also tested as a dimensionality-reduction technique.

Because this is a food-safety classification problem, recall
for the poisonous class is particularly important.
"""
)