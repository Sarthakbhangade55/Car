# ================================================================
# OASIS INFOBYTE DATA SCIENCE INTERNSHIP
# TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING
# ================================================================
# Author: Sarthak Bhangade
# Technology: Python, Pandas, NumPy, Matplotlib, Seaborn,
#             Scikit-learn
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================
# SETTINGS
# ================================================================

warnings.filterwarnings("ignore")

DATA_PATH = "data/car_data.csv"
OUTPUT_DIR = "outputs"

CURRENT_YEAR = pd.Timestamp.today().year

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def print_heading(title):
    """Print a formatted section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def extract_brand(car_name):
    """
    Extract the car brand from Car_Name.

    The function first checks known multi-word/known brands.
    If no brand is found, the first word is used.
    """

    if pd.isna(car_name):
        return "Unknown"

    name = str(car_name).strip().lower()

    known_brands = [
        "mercedes benz",
        "land rover",
        "aston martin",
        "rolls royce",
        "maruti",
        "hyundai",
        "tata",
        "mahindra",
        "toyota",
        "honda",
        "kia",
        "volkswagen",
        "skoda",
        "renault",
        "nissan",
        "mg",
        "jeep",
        "bmw",
        "audi",
        "volvo",
        "lexus",
        "ford",
        "fiat",
        "citroen",
        "byd",
        "isuzu",
        "datsun",
        "chevrolet",
        "mitsubishi",
        "jaguar",
        "porsche",
        "mini",
    ]

    for brand in known_brands:
        if name.startswith(brand):
            return brand.title()

    return name.split()[0].title()


def normalize_category(value):
    """Clean categorical values by removing spaces and standardizing case."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    return value


# ================================================================
# 1. LOAD DATASET
# ================================================================

print_heading("CAR PRICE PREDICTION - DATA LOADING")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at: {DATA_PATH}")
    print("Make sure car_data.csv is inside the data folder.")
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nOriginal Columns:")
print(df.columns.tolist())


# ================================================================
# 2. CLEAN COLUMN NAMES
# ================================================================

df.columns = df.columns.str.strip()

required_columns = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    print(missing_columns)
    raise ValueError("Dataset does not contain the required columns.")


# ================================================================
# 3. INITIAL DATA INSPECTION
# ================================================================

print_heading("INITIAL DATA INSPECTION")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ================================================================
# 4. DATA CLEANING
# ================================================================

print_heading("DATA CLEANING")

# Remove leading/trailing spaces from text columns
text_columns = [
    "Car_Name",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

for column in text_columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )


# Convert numerical columns
numeric_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# Normalize categorical columns
for column in ["Fuel_Type", "Seller_Type", "Transmission"]:
    df[column] = df[column].apply(normalize_category)


# Standardize Fuel_Type
fuel_mapping = {
    "petrol": "Petrol",
    "diesel": "Diesel",
    "cng": "CNG",
    "electric": "Electric",
    "hybrid": "Hybrid",
}

df["Fuel_Type"] = df["Fuel_Type"].map(
    lambda x: fuel_mapping.get(x, x.title() if isinstance(x, str) else x)
)


# Standardize Seller_Type
seller_mapping = {
    "dealer": "Dealer",
    "individual": "Individual",
}

df["Seller_Type"] = df["Seller_Type"].map(
    lambda x: seller_mapping.get(
        x,
        x.title() if isinstance(x, str) else x
    )
)


# Standardize Transmission
transmission_mapping = {
    "manual": "Manual",
    "automatic": "Automatic",
}

df["Transmission"] = df["Transmission"].map(
    lambda x: transmission_mapping.get(
        x,
        x.title() if isinstance(x, str) else x
    )
)


# Display missing values before handling
print("\nMissing values BEFORE handling:")
print(df.isnull().sum())


# Remove duplicate rows
duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after duplicate removal: {len(df)}")


# Drop rows where target is missing
target_missing = df["Selling_Price"].isnull().sum()

if target_missing > 0:
    print(
        f"\nRemoving {target_missing} rows with missing Selling_Price."
    )
    df = df.dropna(subset=["Selling_Price"])


# Fill missing numeric feature values with median
numeric_features_to_fill = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]

for column in numeric_features_to_fill:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())


# Fill missing categorical values with mode
categorical_features_to_fill = [
    "Car_Name",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

for column in categorical_features_to_fill:
    if df[column].isnull().sum() > 0:
        mode_value = df[column].mode()

        if len(mode_value) > 0:
            df[column] = df[column].fillna(mode_value.iloc[0])
        else:
            df[column] = df[column].fillna("Unknown")


print("\nMissing values AFTER handling:")
print(df.isnull().sum())


# ================================================================
# 5. FEATURE ENGINEERING
# ================================================================

print_heading("FEATURE ENGINEERING")

# Create Car_Age
df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Prevent negative car ages
df["Car_Age"] = df["Car_Age"].clip(lower=0)

# Extract Brand from Car_Name
df["Brand"] = df["Car_Name"].apply(extract_brand)

print("\nNew features created:")
print("- Car_Age")
print("- Brand")

print("\nSample engineered data:")
print(
    df[
        [
            "Car_Name",
            "Brand",
            "Year",
            "Car_Age",
            "Selling_Price",
        ]
    ].head(10)
)


# ================================================================
# 6. SAVE CLEANED DATA
# ================================================================

cleaned_data_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_car_data.csv"
)

df.to_csv(cleaned_data_path, index=False)

print(f"\nCleaned dataset saved to: {cleaned_data_path}")


# ================================================================
# 7. EXPLORATORY DATA ANALYSIS - PRICE DISTRIBUTION
# ================================================================

print_heading("EDA - SELLING PRICE DISTRIBUTION")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Selling_Price"],
    bins=30,
    kde=True
)

plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.tight_layout()

price_distribution_path = os.path.join(
    OUTPUT_DIR,
    "01_price_distribution.png"
)

plt.savefig(price_distribution_path, dpi=300)
plt.close()

print(f"Saved: {price_distribution_path}")


# ================================================================
# 8. EDA - PRICE VS FUEL TYPE
# ================================================================

print_heading("EDA - PRICE VS FUEL TYPE")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Fuel_Type",
    y="Selling_Price"
)

plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.tight_layout()

fuel_price_path = os.path.join(
    OUTPUT_DIR,
    "02_price_vs_fuel_type.png"
)

plt.savefig(fuel_price_path, dpi=300)
plt.close()

print(f"Saved: {fuel_price_path}")


# ================================================================
# 9. EDA - PRICE VS CAR AGE
# ================================================================

print_heading("EDA - PRICE VS CAR AGE")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Car_Age",
    y="Selling_Price",
    hue="Fuel_Type",
    alpha=0.75
)

plt.title("Selling Price vs Car Age")
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price")
plt.tight_layout()

car_age_path = os.path.join(
    OUTPUT_DIR,
    "03_price_vs_car_age.png"
)

plt.savefig(car_age_path, dpi=300)
plt.close()

print(f"Saved: {car_age_path}")


# ================================================================
# 10. CORRELATION HEATMAP
# ================================================================

print_heading("CORRELATION HEATMAP")

correlation_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Car_Age",
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "04_correlation_heatmap.png"
)

plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Saved: {heatmap_path}")


# ================================================================
# 11. PREPARE FEATURES AND TARGET
# ================================================================

print_heading("FEATURE AND TARGET PREPARATION")

features = [
    "Car_Age",
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Brand",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

target = "Selling_Price"

X = df[features]
y = df[target]

print("\nFeatures:")
print(features)

print(f"\nTarget: {target}")

print(f"\nFeature shape: {X.shape}")
print(f"Target shape : {y.shape}")


# ================================================================
# 12. TRAIN TEST SPLIT
# ================================================================

print_heading("TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ================================================================
# 13. CATEGORICAL ENCODING
# ================================================================

categorical_features = [
    "Brand",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

numerical_features = [
    "Car_Age",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
    ]
)


# ================================================================
# 14. DEFINE MACHINE LEARNING MODELS
# ================================================================

print_heading("MACHINE LEARNING MODELS")

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
}


# ================================================================
# 15. TRAIN AND EVALUATE MODELS
# ================================================================

print_heading("MODEL TRAINING AND EVALUATION")

results = []
trained_models = {}
predictions_dict = {}

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        }
    )

    trained_models[model_name] = pipeline
    predictions_dict[model_name] = predictions

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


# ================================================================
# 16. MODEL COMPARISON
# ================================================================

print_heading("MODEL COMPARISON")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2_Score",
    ascending=False
).reset_index(drop=True)

print("\nModel Performance:")
print(results_df.to_string(index=False))


# Save model results
results_path = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(f"\nModel results saved to: {results_path}")


# ================================================================
# 17. BEST MODEL
# ================================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_predictions = predictions_dict[best_model_name]

best_mae = results_df.iloc[0]["MAE"]
best_rmse = results_df.iloc[0]["RMSE"]
best_r2 = results_df.iloc[0]["R2_Score"]

print_heading("BEST MODEL")

print(f"Best Model : {best_model_name}")
print(f"MAE        : {best_mae:.2f}")
print(f"RMSE       : {best_rmse:.2f}")
print(f"R² Score   : {best_r2:.4f}")


# ================================================================
# 18. MODEL COMPARISON CHART
# ================================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2_Score"
)

plt.title("Regression Model Comparison - R² Score")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.tight_layout()

model_comparison_path = os.path.join(
    OUTPUT_DIR,
    "05_model_comparison.png"
)

plt.savefig(
    model_comparison_path,
    dpi=300
)

plt.close()

print(f"\nSaved: {model_comparison_path}")


# ================================================================
# 19. FEATURE IMPORTANCE
# ================================================================

print_heading("FEATURE IMPORTANCE")

best_estimator = best_model.named_steps["model"]
best_preprocessor = best_model.named_steps["preprocessor"]

feature_names = (
    best_preprocessor
    .get_feature_names_out()
)

# Remove transformer prefixes
clean_feature_names = []

for name in feature_names:

    name = name.replace(
        "num__",
        ""
    )

    name = name.replace(
        "cat__",
        ""
    )

    clean_feature_names.append(name)


# Tree-based models
if hasattr(best_estimator, "feature_importances_"):

    importances = best_estimator.feature_importances_

# Linear Regression
elif hasattr(best_estimator, "coef_"):

    importances = np.abs(
        best_estimator.coef_
    )

else:

    importances = np.zeros(
        len(clean_feature_names)
    )


feature_importance_df = pd.DataFrame(
    {
        "Feature": clean_feature_names,
        "Importance": importances
    }
)

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nTop 15 Important Features:")
print(
    feature_importance_df.head(15).to_string(
        index=False
    )
)


# Save feature importance
feature_importance_csv = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

feature_importance_df.to_csv(
    feature_importance_csv,
    index=False
)

print(
    f"\nFeature importance saved to: "
    f"{feature_importance_csv}"
)


# ================================================================
# 20. FEATURE IMPORTANCE CHART
# ================================================================

top_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    f"Top Feature Importance - {best_model_name}"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

feature_importance_chart = os.path.join(
    OUTPUT_DIR,
    "06_feature_importance.png"
)

plt.savefig(
    feature_importance_chart,
    dpi=300
)

plt.close()

print(
    f"Saved: {feature_importance_chart}"
)


# ================================================================
# 21. ACTUAL VS PREDICTED PRICES
# ================================================================

print_heading("ACTUAL VS PREDICTED PRICE")

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=y_test,
    y=best_predictions,
    alpha=0.8
)

# Perfect prediction line
minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title(
    f"Actual vs Predicted Selling Price - "
    f"{best_model_name}"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")

plt.tight_layout()

actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "07_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.close()

print(f"Saved: {actual_predicted_path}")


# ================================================================
# 22. SAVE TEST PREDICTIONS
# ================================================================

prediction_df = X_test.copy()

prediction_df["Actual_Selling_Price"] = y_test.values

prediction_df["Predicted_Selling_Price"] = best_predictions

prediction_df["Absolute_Error"] = abs(
    prediction_df["Actual_Selling_Price"]
    - prediction_df["Predicted_Selling_Price"]
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    f"\nTest predictions saved to: "
    f"{prediction_path}"
)


# ================================================================
# 23. GENERATE PROJECT OBSERVATIONS
# ================================================================

print_heading("GENERATING PROJECT OBSERVATIONS")

average_price = df["Selling_Price"].mean()
highest_price = df["Selling_Price"].max()
lowest_price = df["Selling_Price"].min()

average_age = df["Car_Age"].mean()

most_common_fuel = (
    df["Fuel_Type"]
    .value_counts()
    .idxmax()
)

most_common_transmission = (
    df["Transmission"]
    .value_counts()
    .idxmax()
)

top_feature = (
    feature_importance_df.iloc[0]["Feature"]
)

observations = f"""
============================================================
CAR PRICE PREDICTION - PROJECT OBSERVATIONS
============================================================

Dataset Information
--------------------
Total Records       : {len(df)}
Total Features      : {len(df.columns)}
Current Year Used   : {CURRENT_YEAR}

Price Analysis
--------------
Average Selling Price : {average_price:.2f}
Highest Selling Price : {highest_price:.2f}
Lowest Selling Price  : {lowest_price:.2f}

Average Car Age
---------------
{average_age:.2f} years

Most Common Fuel Type
---------------------
{most_common_fuel}

Most Common Transmission
------------------------
{most_common_transmission}

Machine Learning Results
------------------------
Best Model : {best_model_name}

MAE        : {best_mae:.2f}
RMSE       : {best_rmse:.2f}
R2 Score   : {best_r2:.4f}

Most Important Feature
----------------------
{top_feature}

Models Compared
---------------
1. Linear Regression
2. Random Forest
3. Gradient Boosting

EDA Visualizations
------------------
1. Selling Price Distribution
2. Selling Price vs Fuel Type
3. Selling Price vs Car Age
4. Correlation Heatmap
5. Model Comparison
6. Feature Importance
7. Actual vs Predicted Price

Conclusion
----------
The machine learning models were trained to predict the selling
price of used cars using car age, present price, kilometers driven,
owner information, brand, fuel type, seller type and transmission.

The best performing model was selected using R2 Score while MAE
and RMSE were also considered for evaluation.

============================================================
Author: Sarthak Bhangade
Oasis Infobyte Data Science Internship
Task 3 - Car Price Prediction
============================================================
"""

observations_path = os.path.join(
    OUTPUT_DIR,
    "observations.txt"
)

with open(
    observations_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(observations)

print(
    f"Observations saved to: "
    f"{observations_path}"
)


# ================================================================
# 24. FINAL OUTPUT SUMMARY
# ================================================================

print_heading("PROJECT COMPLETED SUCCESSFULLY")

print("""
Generated Files:
------------------------------------------------------------
outputs/
│
├── cleaned_car_data.csv
├── model_results.csv
├── feature_importance.csv
├── test_predictions.csv
├── observations.txt
│
├── 01_price_distribution.png
├── 02_price_vs_fuel_type.png
├── 03_price_vs_car_age.png
├── 04_correlation_heatmap.png
├── 05_model_comparison.png
├── 06_feature_importance.png
└── 07_actual_vs_predicted.png
------------------------------------------------------------
""")

print("Best Model:", best_model_name)
print(f"MAE      : {best_mae:.2f}")
print(f"RMSE     : {best_rmse:.2f}")
print(f"R² Score : {best_r2:.4f}")

print("\nTask 3 completed successfully! 🚗🤖")
print("Author: Sarthak Bhangade")