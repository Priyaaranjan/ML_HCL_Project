import pandas as pd

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

print("=" * 60)
print("MUSHROOM DATASET ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(data.shape)


# ============================================================
# 3. FIRST FIVE ROWS
# ============================================================

print("\nFirst 5 Records:")
print(data.head())


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\nData Types:")
print(data.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 6. '?' VALUES
# ============================================================

print("\nUnknown '?' Values:")

for column in data.columns:
    count = (data[column] == "?").sum()

    if count > 0:
        print(column, ":", count)


# ============================================================
# 7. DUPLICATE RECORDS
# ============================================================

print("\nDuplicate Records:")

duplicate_count = data.duplicated().sum()

print(duplicate_count)


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print("\nTarget Distribution:")

print(data["class"].value_counts())

print("\nTarget Percentage:")

print(data["class"].value_counts(normalize=True) * 100)


# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print("\nUnique Values Per Feature:")

for column in data.columns:
    print(
        column,
        ":",
        data[column].nunique(),
        "unique values"
    )


# ============================================================
# 10. VALUE FREQUENCY
# ============================================================

print("\nFeature Value Frequencies:")

for column in data.columns:

    print("\n------------------------------")
    print(column)
    print("------------------------------")

    print(data[column].value_counts())


# ============================================================
# 11. DATASET INFORMATION
# ============================================================

print("\nDataset Information:")

data.info()


# ============================================================
# 12. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary:")

print(data.describe(include="all"))


# ============================================================
# 13. OUTLIER CHECK
# ============================================================

print("\nOutlier Investigation:")

print(
    "The UCI Mushroom dataset contains categorical features."
)

print(
    "Traditional numerical outlier detection such as IQR "
    "is not directly applicable to these categorical features."
)


# ============================================================
# 14. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

print("Rows:", data.shape[0])
print("Columns:", data.shape[1])
print("Duplicates:", duplicate_count)

unknown_values = (data == "?").sum().sum()

print("Unknown '?' values:", unknown_values)

print("\nTarget classes:")
print(data["class"].value_counts())

print("\nAnalysis completed.")