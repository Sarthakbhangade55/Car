# Car Price Prediction with Machine Learning


# 📌 Project Objective

The main objective of this project is to build and compare Machine Learning regression models that can accurately predict the selling price of a used car.

The project includes:

* Data loading and inspection
* Data cleaning
* Missing value handling
* Duplicate removal
* Categorical value cleaning
* Feature engineering
* Exploratory Data Analysis
* Data visualization
* Feature encoding
* Correlation analysis
* Train-test splitting
* Training multiple regression models
* Model evaluation using MAE, RMSE, and R² Score
* Best model selection
* Feature importance analysis
* Actual vs Predicted price comparison

---

# 📊 Dataset

The project uses a CarDekho-style used car dataset.

The dataset contains the following features:

| Feature       | Description                       |
| ------------- | --------------------------------- |
| Car_Name      | Name of the car                   |
| Year          | Manufacturing year                |
| Selling_Price | Selling price of the car          |
| Present_Price | Current/original price of the car |
| Kms_Driven    | Total kilometers driven           |
| Fuel_Type     | Type of fuel used                 |
| Seller_Type   | Dealer or Individual              |
| Transmission  | Manual or Automatic               |
| Owner         | Number of previous owners         |

The target variable is:

```text
Selling_Price
```

---

# 🧹 Data Cleaning

The dataset was cleaned before training the Machine Learning models.

The following operations were performed:

* Checked dataset shape
* Checked column names
* Checked data types
* Checked missing values
* Removed duplicate records
* Cleaned inconsistent categorical values
* Handled missing numerical values
* Handled missing categorical values

Examples of categorical cleaning:

```text
petrol
Petrol
 PETROL
```

All inconsistent values were standardized into:

```text
Petrol
```

The same cleaning process was applied to Fuel Type, Seller Type, and Transmission.

---

# 🔧 Feature Engineering

Two important features were created.

## Car Age

Car age was calculated using:

```text
Car_Age = Current Year - Year
```

This feature helps the model understand how the age of the vehicle affects its selling price.

## Brand Extraction

The car brand was extracted from the `Car_Name` column.

Examples:

```text
Maruti Swift → Maruti
Hyundai i20 → Hyundai
Honda City → Honda
Toyota Corolla → Toyota
```

The extracted brand was used as an additional feature for model training.

---

# 📈 Exploratory Data Analysis

Several visualizations were created to understand the dataset and identify important patterns.

The following analysis was performed:

### 1. Selling Price Distribution

A histogram was created to understand how used car prices are distributed.

Output:

```text
01_price_distribution.png
```

### 2. Selling Price vs Fuel Type

A box plot was created to compare the selling prices of cars with different fuel types.

Output:

```text
02_price_vs_fuel_type.png
```

### 3. Selling Price vs Car Age

A scatter plot was created to analyze the relationship between car age and selling price.

Output:

```text
03_price_vs_car_age.png
```

### 4. Correlation Heatmap

A correlation heatmap was generated to understand the relationship between numerical features.

Output:

```text
04_correlation_heatmap.png
```

---

# 🔤 Feature Encoding

Machine Learning models require numerical input.

Categorical features were converted into numerical form using **One-Hot Encoding**.

The following features were encoded:

* Brand
* Fuel_Type
* Seller_Type
* Transmission

The encoder was configured to handle unknown categories safely.

---

# 🧪 Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

The split was performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

# 🤖 Machine Learning Models

The following regression models were trained and compared.

## 1. Linear Regression

Linear Regression was used as a baseline model.

It attempts to find a linear relationship between the input features and the selling price.

## 2. Random Forest Regressor

Random Forest uses multiple decision trees to make predictions.

It is useful for identifying nonlinear relationships between car features and selling prices.

## 3. Gradient Boosting Regressor

Gradient Boosting builds multiple models sequentially and improves predictions by reducing previous errors.

---

# 📏 Model Evaluation

The models were evaluated using the following metrics.

## Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted selling prices.

Lower MAE indicates better performance.

## Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction errors.

Lower RMSE indicates better performance.

## R² Score

R² Score measures how well the model explains the variation in car selling prices.

Higher R² Score indicates better performance.

---

# 🏆 Model Comparison

The following models were compared:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The model evaluation results are automatically saved in:

```text
outputs/model_results.csv
```

The best-performing model is selected automatically based on its performance.

---

# ⭐ Feature Importance

Feature importance analysis was performed for the best-performing model.

This analysis helps identify which features have the greatest impact on predicting car selling prices.

The results are saved as:

```text
outputs/feature_importance.csv
```

A feature importance visualization is also generated:

```text
outputs/06_feature_importance.png
```

---

# 📊 Visualizations Preview

The project automatically generates the following visualizations:

```text
01_price_distribution.png
```

Selling price distribution of used cars.

```text
02_price_vs_fuel_type.png
```

Comparison of car selling prices based on fuel type.

```text
03_price_vs_car_age.png
```

Relationship between car age and selling price.

```text
04_correlation_heatmap.png
```

Correlation between numerical features.

```text
05_model_comparison.png
```

Comparison of Machine Learning model performance.

```text
06_feature_importance.png
```

Most important features affecting car selling price.

```text
07_actual_vs_predicted.png
```

Comparison between actual and predicted selling prices.

---

# 📁 Project Structure

```text
Car-Price-Prediction/
│
├── data/
│   └── car_data.csv
│
├── src/
│   └── car_price_prediction.py
│
├── outputs/
│   ├── cleaned_car_data.csv
│   ├── model_results.csv
│   ├── feature_importance.csv
│   ├── test_predictions.csv
│   ├── observations.txt
│   ├── 01_price_distribution.png
│   ├── 02_price_vs_fuel_type.png
│   ├── 03_price_vs_car_age.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_model_comparison.png
│   ├── 06_feature_importance.png
│   └── 07_actual_vs_predicted.png
│
├── README.md
│
└── requirements.txt
```

---

# 🛠️ Technologies Used

The following technologies and libraries were used:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Visual Studio Code
* GitHub

---

# 📦 Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 📄 Requirements

Create a file named:

```text
requirements.txt
```

Add the following:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# ▶️ How to Run the Project

## Step 1: Open the Project

Open the project folder in Visual Studio Code.

## Step 2: Open Terminal

Go to:

```text
Terminal → New Terminal
```

## Step 3: Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

## Step 4: Run the Python Program

Run:

```bash
python src/car_price_prediction.py
```

## Step 5: Check the Results

After successful execution, all generated files will be available inside the:

```text
outputs/
```

folder.

---

# 🔄 Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Cleaning
   │
   ├── Missing Value Handling
   ├── Duplicate Removal
   └── Categorical Data Cleaning
   │
   ▼
Feature Engineering
   │
   ├── Car Age
   └── Brand Extraction
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Encoding
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ├── Linear Regression
   ├── Random Forest Regressor
   └── Gradient Boosting Regressor
   │
   ▼
Model Evaluation
   │
   ├── MAE
   ├── RMSE
   └── R² Score
   │
   ▼
Best Model Selection
   │
   ▼
Feature Importance
   │
   ▼
Prediction and Results
```

---

# 📚 Key Learning Outcomes

Through this project, the following concepts were implemented and learned:

* Data preprocessing
* Missing value handling
* Duplicate removal
* Categorical data cleaning
* Feature engineering
* Exploratory Data Analysis
* Data visualization
* One-Hot Encoding
* Regression algorithms
* Train-test splitting
* Machine Learning model evaluation
* MAE calculation
* RMSE calculation
* R² Score analysis
* Model comparison
* Feature importance analysis
* Predictive analytics

---

# 💡 Conclusion

This project demonstrates a complete Machine Learning workflow for predicting used car selling prices.

The dataset was cleaned and preprocessed before creating new features such as **Car Age** and **Brand**. Exploratory Data Analysis was performed to understand the relationships between vehicle characteristics and selling prices.

Three regression models were trained:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The models were evaluated using MAE, RMSE, and R² Score. The best-performing model was automatically identified based on its evaluation performance.

The project also generated feature importance analysis, prediction results, model comparison results, and multiple visualizations.

This project provides practical experience in **Python, Data Analysis, Data Preprocessing, Machine Learning Regression, Model Evaluation, and Predictive Analytics**.

---

# 🚀 Future Improvements

The project can be improved further by:

* Using a larger real-world dataset
* Adding more vehicle features
* Performing hyperparameter tuning
* Using cross-validation
* Implementing XGBoost
* Creating a Streamlit web application
* Deploying the trained model
* Saving the trained model using Joblib
* Creating an API for real-time car price prediction

---

# 👨‍💻 Author

**Sarthak Bhangade**

B.Tech – Artificial Intelligence & Data Science
Sanjivani University
2024 – 2028

**Oasis Infobyte Data Science Internship**

### Task 3 – Car Price Prediction with Machine Learning

---

# ✅ Project Status

```text
Dataset Preparation              ✅ Completed
Data Cleaning                    ✅ Completed
Missing Value Handling           ✅ Completed
Duplicate Removal                ✅ Completed
Feature Engineering              ✅ Completed
Exploratory Data Analysis        ✅ Completed
Data Visualization               ✅ Completed
Categorical Encoding             ✅ Completed
Correlation Analysis             ✅ Completed
Train-Test Split                 ✅ Completed
Linear Regression                ✅ Completed
Random Forest Regression         ✅ Completed
Gradient Boosting Regression     ✅ Completed
MAE Evaluation                   ✅ Completed
RMSE Evaluation                  ✅ Completed
R² Score Evaluation              ✅ Completed
Best Model Selection             ✅ Completed
Feature Importance Analysis      ✅ Completed
Prediction Generation            ✅ Completed
Project Documentation            ✅ Completed

PROJECT STATUS: COMPLETED ✅
```

---

# 📜 License

This project was developed for educational and internship purposes as part of the Oasis Infobyte Data Science Internship.

---

👨‍💻 **Author: Sarthak Bhangade**
