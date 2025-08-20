# =============================================================================
# Streamlit App for Loan Default Prediction with Explanations and Prediction Logging
# =============================================================================

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. Load the saved pipeline and dataset for dynamic inputs
# =============================================================================

@st.cache_resource
def load_model():
    # Load the trained pipeline (preprocessor + model)
    return joblib.load("final_pipeline_with_smote_gridsearch.pkl")

@st.cache_resource
def load_data():
    # Load original dataset to extract feature stats
    return pd.read_csv("BankChurners.csv")

# Load model pipeline and dataset once, cache results for efficiency
pipeline = load_model()
df = load_data()

# =============================================================================
# 2. Set up the Streamlit app title and description
# =============================================================================

st.title("Customer Churn Prediction App")

# Introduction text explaining the app purpose
st.write("""
This app predicts whether a customer will close their accounts based on user inputs.  
The model was trained on banking data and includes preprocessing steps.  
Adjust inputs on the left sidebar and click 'Predict' to see the results.
""")

# =============================================================================
# 3. Extract feature info for creating input widgets dynamically
# =============================================================================

# Extract the preprocessor step (a ColumnTransformer)
preprocessor = pipeline.named_steps['preprocessor']

# Extract numeric and categorical feature names used in the pipeline
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]

# Get the fitted OneHotEncoder to extract categorical feature categories
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_categories = ohe.categories_

# Build a dictionary mapping each categorical feature to its list of categories
cat_feature_categories = dict(zip(categorical_features, cat_categories))

# =============================================================================
# 4. Create user input widgets dynamically based on feature types and data stats
# =============================================================================

st.sidebar.header("Input Features")

user_input = {}

# For numeric features, create number input sliders with dataset min, max, and mean as defaults
for feature in numeric_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    user_input[feature] = st.sidebar.number_input(
        label=f"{feature} (numeric)",
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=0.01
    )

# For categorical features, create dropdown menus with categories observed during training
for feature in categorical_features:
    categories = cat_feature_categories[feature]
    default_index = 0  # default to first category
    user_input[feature] = st.sidebar.selectbox(
        label=f"{feature} (categorical)",
        options=categories,
        index=default_index
    )

# =============================================================================
# 5. Initialize or retrieve a session state list to store all predictions made during the session
# =============================================================================

if 'predictions_log' not in st.session_state:
    st.session_state['predictions_log'] = []

# =============================================================================
# 6. When Predict button clicked, process input and show prediction and save it
# =============================================================================

if st.sidebar.button("Predict"):
    # Convert input dictionary to single-row DataFrame to feed into the pipeline
    input_df = pd.DataFrame([user_input])

    # Run prediction and get probability for positive class (default=1)
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    # Show prediction result with conditional formatting
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("Predicted Churn: YES")
    else:
        st.success("Predicted Churn: NO")

    st.write(f"Model confidence (probability of default): {proba:.2%}")

    # Log the prediction along with inputs and probability into session state
    record = user_input.copy()
    record['Prediction'] = 'YES' if prediction == 1 else 'NO'
    record['Churn Probability'] = round(proba, 4)
    st.session_state['predictions_log'].append(record)

# =============================================================================
# 7. Show feature importance if supported by the model
# =============================================================================

model = pipeline.named_steps['model']

if hasattr(model, 'feature_importances_'):
    # Compose feature names after preprocessing (including one-hot encoded categorical features)
    feature_names = list(numeric_features)
    if categorical_features:
        ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
        feature_names += ohe_feature_names

    # Extract feature importances
    importances = model.feature_importances_

    # Create a DataFrame to hold feature names and their importance scores
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(15)

    # Plot top 15 feature importances using seaborn barplot
    st.subheader("Top 15 Feature Importances")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
else:
    # If the model has no feature_importances_ attribute (e.g., Logistic Regression), inform the user
    st.info("Feature importance is not available for the selected model.")

# =============================================================================
# 8. Display a table of all predictions made in this session for user reference
# =============================================================================

if st.session_state['predictions_log']:
    st.subheader("Prediction History in This Session")
    # Convert list of dicts to DataFrame for pretty display
    history_df = pd.DataFrame(st.session_state['predictions_log'])
    st.dataframe(history_df)


# After displaying prediction history
def history_df():
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction History", csv, "prediction_history.csv", "text/csv")

    st.subheader("Batch Prediction: Upload CSV or Excel File")
uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)
        
        # Ensure columns match expected features
        expected_cols = list(numeric_features) + list(categorical_features)
        missing_cols = [col for col in expected_cols if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
        else:
            # Predict
            preds = pipeline.predict(batch_df[expected_cols])
            probas = pipeline.predict_proba(batch_df[expected_cols])[:, 1]
            batch_df['Predicted Churn'] = np.where(preds == 1, 'YES', 'NO')
            batch_df['Churn Probability'] = np.round(probas, 4)
            st.success("Batch prediction completed!")
            st.dataframe(batch_df)

            # Download results
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "batch_predictions.csv", "text/csv")
    except Exception as e:

        st.error(f"Error processing file: {e}")

