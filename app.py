import streamlit as st
import numpy as np
import joblib

# Load model
xgb = joblib.load('loan_approval_model.pkl')  # pastikan nama file sesuai

# Header HTML
html_temp = """
<div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">Loan Eligibility Prediction App</h1> 
    <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
</div>
"""

# Deskripsi App
desc_temp = """ ### Loan Prediction App 
This app is used by Credit team for deciding Loan Application
"""

# Mapping untuk encoding
property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# Fungsi Prediksi
def predict(inputs):
    # Feature engineering
    applicant_income = inputs['Applicant_Income']
    coapplicant_income = inputs['Coapplicant_Income']
    total_income = applicant_income + coapplicant_income
    loan_income_ratio = inputs['Loan_Amount'] / total_income if total_income > 0 else 0

    # Encode categorical
    gen = 0 if inputs['Gender'] == "Male" else 1
    mar = 0 if inputs['Married'] == "Yes" else 1
    edu = 0 if inputs['Education'] == "Graduate" else 1
    sem = 0 if inputs['Self_Employed'] == "Yes" else 1
    pro = property_map[inputs['Property_Area']]

    # Gabung semua fitur sesuai training
    feature_array = np.array([[gen, mar, int(inputs['Dependents']), edu, sem,
                               applicant_income, coapplicant_income,
                               inputs['Loan_Amount'], inputs['Loan_Amount_Term'],
                               int(inputs['Credit_History']), pro,
                               total_income, loan_income_ratio]])
    
    prediction = xgb.predict(feature_array)
    return 'Eligible' if prediction[0] == 1 else 'Not Eligible'

# ML App Interface
def run_ml_app():
    st.markdown(html_temp, unsafe_allow_html=True)

    st.subheader("Loan Eligibility Prediction")
    
    # Input form
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self_Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant_Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant_Income", min_value=0)
    loan_amount = st.number_input("Loan_Amount", min_value=0)
    loan_amount_term = st.number_input("Loan_Amount_Term", min_value=10, max_value=360)
    credit_history = st.selectbox("Credit_History", ["1", "0"])
    property_area = st.selectbox("Property_Area", ["Rural", "Semiurban", "Urban"])

    inputs = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents if dependents != "3+" else 3,
        'Education': education,
        'Self_Employed': self_employed,
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    if st.button("Predict"):
        result = predict(inputs)
        if result == 'Eligible':
            st.success(f"✅ You are {result} for the loan")
        else:
            st.error(f"❌ You are {result} for the loan")

# Main App
def main():
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown(html_temp, unsafe_allow_html=True)
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

if __name__ == "__main__":
    main()
