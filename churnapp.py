# =============================================================================
# Streamlit App for Loan Default Prediction with Explanations and Prediction Logging
# =============================================================================
# =============================================================================
# Modern Streamlit App for Customer Churn Prediction (Enhanced UI)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# =============================================================================
# 1. Set page config and add logo/banner
# =============================================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="customer-churn-icon-design-vector.jpg",
    layout="wide"
)

col1, col2 = st.columns([1, 8])
with col1:
    st.image("customer-churn-icon-design-vector.jpg", width=80)
with col2:
    st.markdown(
        """
        <h1 style='color:#4B8BBE; margin-bottom:0;'>Customer Churn Prediction App</h1>
        <p style='color:#555; font-size:18px; margin-top:0;'>Predict customer churn with confidence using machine learning.</p>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 2. Load the saved pipeline and dataset for dynamic inputs
# =============================================================================

@st.cache_resource
def load_model():
    return joblib.load("final_pipeline_with_smote_gridsearch.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("BankChurners.csv")

pipeline = load_model()
df = load_data()

# =============================================================================
# 3. Extract feature info for creating input widgets dynamically
# =============================================================================

preprocessor = pipeline.named_steps['preprocessor']
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_categories = ohe.categories_
cat_feature_categories = dict(zip(categorical_features, cat_categories))

# =============================================================================
# 4. Sidebar: Navigation
# =============================================================================

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Single Prediction", "Batch Prediction", "About"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Powered by EMPIRE Data Science Team**")

# =============================================================================
# 5. Single Prediction Page
# =============================================================================

if page == "Single Prediction":
    st.header("🔎 Single Customer Prediction")
    with st.expander("ℹ️ How to use this section", expanded=False):
        st.write(
            "Fill in the customer details in the sidebar and click **Predict** to see if the customer is likely to churn."
        )

    if 'predictions_log' not in st.session_state:
        st.session_state['predictions_log'] = []

    user_input = {}
    with st.form("single_pred_form"):
        st.subheader("Enter Customer Details")
        for feature in numeric_features:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            user_input[feature] = st.number_input(
                label=f"{feature} (numeric)",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=0.01,
                help=f"Enter a value between {min_val:.2f} and {max_val:.2f} (mean: {mean_val:.2f})"
            )

        for feature in categorical_features:
            categories = cat_feature_categories[feature]
            default_index = 0
            user_input[feature] = st.selectbox(
                label=f"{feature} (categorical)",
                options=categories,
                index=default_index,
                help=f"Select the appropriate category for {feature}."
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("Predicted Churn: YES")
        else:
            st.success("Predicted Churn: NO")
        st.write(f"Model confidence (probability of churn): {proba:.2%}")

        record = user_input.copy()
        record['Prediction'] = 'YES' if prediction == 1 else 'NO'
        record['Churn Probability'] = round(proba, 4)
        st.session_state['predictions_log'].append(record)

    # Feature importance
    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        feature_names = list(numeric_features)
        if categorical_features:
            ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))
            feature_names += ohe_feature_names
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
        with st.expander("🔬 Top 15 Feature Importances", expanded=False):
            fig, ax = plt.subplots(figsize=(8,6))
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Feature importance is not available for the selected model.")

    # Prediction history and download
    if st.session_state['predictions_log']:
        with st.expander("🕑 Prediction History in This Session", expanded=False):
            history_df = pd.DataFrame(st.session_state['predictions_log'])
            st.dataframe(history_df)
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Prediction History", csv, "prediction_history.csv", "text/csv")

# =============================================================================
# 6. Batch Prediction Page
# =============================================================================

elif page == "Batch Prediction":
    st.header("📂 Batch Prediction: Upload CSV or Excel File")
    with st.expander("ℹ️ How to use this section", expanded=False):
        st.write(
            "Upload a CSV or Excel file containing customer data. The app will predict churn for each row."
        )
        st.markdown("**Sample Input Template:**")
        sample_df = df[list(numeric_features) + list(categorical_features)].head(2)
        st.dataframe(sample_df)

    uploaded_file = st.file_uploader("Upload your input file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
            expected_cols = list(numeric_features) + list(categorical_features)
            missing_cols = [col for col in expected_cols if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
            else:
                progress = st.progress(0, text="Running batch prediction...")
                preds = []
                probas = []
                for i in range(len(batch_df)):
                    row = batch_df.iloc[[i]][expected_cols]
                    pred = pipeline.predict(row)[0]
                    proba = pipeline.predict_proba(row)[0][1]
                    preds.append(pred)
                    probas.append(proba)
                    progress.progress((i+1)/len(batch_df), text=f"Processing row {i+1}/{len(batch_df)}")
                batch_df['Predicted Churn'] = np.where(np.array(preds) == 1, 'YES', 'NO')
                batch_df['Churn Probability'] = np.round(probas, 4)
                progress.empty()
                st.success("Batch prediction completed!")
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv, "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# =============================================================================
# 7. About Page
# =============================================================================

elif page == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
    This web application predicts customer churn using a machine learning model trained on real-world banking data.
    
    **Features:**
    - Predict churn for individual customers or in batches.
    - Visualize feature importance.
    - Download prediction results.
    - Transparent model evaluation.
    """)
    st.markdown("**Dataset Sample:**")
    st.dataframe(df.head(5))

    st.markdown("**Churn Distribution in Training Data:**")
    fig, ax = plt.subplots()
    sns.countplot(x='Attrition_Flag', data=df, ax=ax)
    st.pyplot(fig)

    # Model transparency: ROC and confusion matrix
    st.markdown("**Model Performance (Training Data):**")
    X = df[list(numeric_features) + list(categorical_features)]
    y = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    st.markdown("**ROC Curve:**")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Confusion Matrix
    st.markdown("**Confusion Matrix:**")
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Contact:** empire.datascience@example.com")

# =============================================================================
# 8. Footer
# =============================================================================

st.markdown(
    """
    <hr>
    <div style='text-align:center; color: #888;'>
        &copy; 2025 EMPIRE Data Science Team | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
