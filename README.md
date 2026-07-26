# Universal ML Pipeline

A flexible, end-to-end Machine Learning pipeline built with Python, Pandas, and Scikit-Learn. This project provides a simple and reusable framework for loading raw data, preprocessing features, splitting datasets, training models, evaluating performance, and selecting the best model for both Classification and Regression tasks.

## ✨ Features

- Automated preprocessing
  - Detects numerical and categorical columns automatically
  - Handles missing values using median for numeric data and most frequent values for categorical data
  - Applies scaling for numeric features and encoding for categorical features

- Flexible data splitting
  - Supports train-test splitting with optional stratification for classification problems

- Multi-model evaluation
  - Classification models: Logistic Regression, Random Forest Classifier
  - Regression models: Linear Regression, Random Forest Regressor

- Best model selection
  - Evaluates models using accuracy for classification and RMSE for regression
  - Automatically stores the best-performing model

## 🛠️ Project Structure

```text
Universal_ML_Template/
├── Universal_ML_Template.ipynb   # Main notebook containing the ML pipeline
├── README.md                      # Project documentation
└── (optional) heart_disease.csv   # Sample dataset
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the required dependencies:

```bash
pip install pandas numpy scikit-learn
```

### Basic Usage

Below is a quick example of how to initialize and run the pipeline:

```python
from pipeline import UniversalMLPipeline

if __name__ == "__main__":
    DATA_PATH = "heart_disease.csv"
    TARGET_COLUMN = "Disease"
    PROBLEM = "Classification"  # Options: "Classification" or "Regression"

    pipeline = UniversalMLPipeline(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        problem_type=PROBLEM,
    )

    try:
        pipeline.load_data()
        pipeline.split_data(test_size=0.2, random_state=42)
        pipeline.model_train_evaluting()
    except FileNotFoundError:
        print("Error: Dataset file not found at the specified path.")
```

> If your implementation is stored inside the notebook instead of a Python file, adapt the usage accordingly.

## ⚙️ How It Works

1. Data Loading
   - Reads the CSV file using Pandas
   - Displays the dataset head to confirm the data is loaded correctly

2. Preprocessing and Splitting
   - Separates features and target variable
   - Builds a preprocessing pipeline using:
     - SimpleImputer + StandardScaler for numeric features
     - SimpleImputer + OneHotEncoder for categorical features
   - Splits the data into train and test sets while preventing data leakage

3. Model Training and Evaluation
   - Selects appropriate models based on the problem type
   - Trains each model and evaluates its performance
   - Chooses the best model based on the defined metric

## 📊 Example Output

```text
Data Loading.....
--- LogisticRegression Training ---
LogisticRegression Accuracy: 0.8525

--- RandomForestClassifier Training ---
RandomForestClassifier Accuracy: 0.8852

Best model is: RandomForestClassifier
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License.
