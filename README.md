# Customer Churn Prediction App

This project is a **Streamlit web application** that predicts whether a customer will churn (close their account) based on their banking data. The app uses a machine learning pipeline trained on real-world customer data and provides both single and batch prediction capabilities.

---

## Features

- **Interactive User Interface:** Enter customer features via sidebar widgets and get instant churn predictions.
- **Batch Prediction:** Upload a CSV or Excel file to predict churn for multiple customers at once.
- **Prediction History:** View and download all predictions made during your session.
- **Feature Importance:** Visualize which features are most influential in the model’s decisions.
- **Downloadable Results:** Download your prediction history or batch results as CSV files.

---

## How to Run

1. **Clone the repository** and navigate to the project folder:
    ```bash
    git clone <your-repo-url>
    cd CHURN
    ```

2. **Install dependencies** (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure the following files are present in the project directory:**
    - `churnapp.py` (the Streamlit app)
    - `final_pipeline_with_smote_gridsearch.pkl` (the trained ML pipeline)
    - `BankChurners.csv` (the dataset for extracting feature statistics)

4. **Run the Streamlit app:**
    ```bash
    streamlit run churnapp.py
    ```

5. **Open your browser** to the provided local URL (usually http://localhost:8501).

---

## Usage

- **Single Prediction:**  
  Use the sidebar to enter customer details and click **Predict**. The app will display the churn prediction and model confidence.

- **Batch Prediction:**  
  Upload a CSV or Excel file with the required columns. The app will predict churn for each row and let you download the results.

- **Prediction History:**  
  All predictions in your session are shown in a table and can be downloaded as a CSV.

---

## Input File Format for Batch Prediction

Your CSV or Excel file should contain the following columns (matching the model’s features):

- All numeric features (e.g., `Customer_Age`, `Credit_Limit`, etc.)
- All categorical features (e.g., `Gender`, `Education_Level`, etc.)

**Column names must match those used during training.**

---

## Requirements

- Python 3.8+
- See `requirements.txt` for full package list (includes `streamlit`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, etc.)

---

## Notes

- The app uses a pre-trained pipeline (`final_pipeline_with_smote_gridsearch.pkl`) that includes all preprocessing steps.
- For best results, ensure your input data matches the expected format and categories.

---

## License

This project is for educational and demonstration purposes.

---

## Acknowledgements

- Dataset: [BankChurners.csv](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- Built with [Streamlit](https://streamlit.io/)

---
