# House Rent Prediction Project

This project aims to predict monthly house rent prices in major Indian cities using supervised machine learning regression models. The final solution is deployed as an interactive web application using Streamlit.

---

## Dataset Description

- **Source:** Kaggle - House Rent Dataset
- **Total Records:** Approximately 4700 entries
- **Missing Values:** Few missing values were found and dropped
- **Outliers:** Detected and removed using the IQR method

### Selected Features for Modeling:

| Column Name            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| BHK                    | Number of bedrooms in the house                                             |
| Size                   | Area of the house in square feet                                            |
| Area Type              | Type of area: Super Area, Carpet Area, or Built Area                        |
| Furnishing Status      | Whether the property is Furnished, Semi-Furnished, or Unfurnished           |
| Bathroom               | Number of bathrooms                                                         |
| Month                  | Month in which the house was posted for rent                                |
| Floor_Num              | Floor number of the property (e.g., 0 = Ground, 1 = 1st floor, etc.)        |
| Apartment_num          | Total number of apartments in the Floor                                 |
| City                   | City name: Bangalore, Chennai, Delhi, Hyderabad, Kolkata, or Mumbai         |
| Tenant Preferred       | Type of tenant preferred: Bachelors, Family, or Both                        |
| Point of Contact       | How to contact the advertiser: Agent, Builder, or Owner                     |

All categorical features were encoded using either Label Encoding or One-Hot Encoding as appropriate.

---

## Data Preprocessing Steps

- Removed null values
- Removed outliers using the IQR method
- Converted Month to numerical format
- Applied Label Encoding on:
  - Area Type
  - Furnishing Status
- Applied OneHot Encoding on:
  - City
  - Tenant Preferred
  - Point of Contact
- Applied feature scaling using `StandardScaler` for selected numeric columns

---

## Machine Learning Models

Several regression models were trained and evaluated, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor
- XGBoost Random Forest Regressor

Each model was tuned using `GridSearchCV` with cross-validation (cv=3) to find the best hyperparameters.

---

## Best Performing Model

After hyperparameter tuning, the **Gradient Boosting Regressor** achieved the highest R² score on the validation set. This model was selected as the final model and saved using pickle.

---

## Streamlit Web Application

A Streamlit app was developed to provide an easy-to-use interface where users can input property details and get a real-time prediction of the estimated rent.

### Technologies Used:

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- scikit-learn
- Gradient boosting
- Streamlit

---

## Repository Contents

- `app.py`: The Streamlit application
- `model.pkl`: The trained Gradient Boosting Regressor
- `encoder.pkl`: OneHotEncoder used in preprocessing
- `scaler.pkl`: StandardScaler used in preprocessing
- `requirements.txt`: List of required Python packages
- `README.md`: Project overview and instructions

---

## Running the App Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/house-rent-predictor.git
cd house-rent-predictor
