import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and data
model = joblib.load("knn_model.pkl")
accounts_df = pd.read_csv("accounts.csv")
alerts_df = pd.read_csv("alerts.csv")

# Constants
amount_bins = [0, 100, 500, 1000, 5000, np.inf]
amount_labels = ['0-100', '100-500', '500-1000', '1000-5000', '5000+']
threshold = 0.4

# Helper functions
def get_account_info(account_id, prefix):
    info = accounts_df[accounts_df['ACCOUNT_ID'] == account_id]
    if not info.empty:
        return {
            f"{prefix}_COUNTRY": info.iloc[0]['COUNTRY'],
            f"{prefix}_ACCOUNT_TYPE": info.iloc[0]['ACCOUNT_TYPE']
        }
    return {
        f"{prefix}_COUNTRY": "UNKNOWN",
        f"{prefix}_ACCOUNT_TYPE": "UNKNOWN"
    }

def get_alerts(account_id):
    alerts = alerts_df[alerts_df['ACCOUNT_ID'] == account_id]
    if not alerts.empty:
        return alerts[['ALERT_TYPE', 'ALERT_DESC']].to_dict(orient='records')
    return []

def make_prediction(input_data):
    input_data['AMOUNT_BIN'] = pd.cut([input_data['TX_AMOUNT']], bins=amount_bins, labels=amount_labels)[0]
    input_df = pd.DataFrame([input_data])
    input_df = input_df[[
        'TX_AMOUNT', 'HOUR', 'SENDER_COUNTRY', 'RECEIVER_COUNTRY',
        'SENDER_ACCOUNT_TYPE', 'RECEIVER_ACCOUNT_TYPE', 'AMOUNT_BIN'
    ]]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return prediction, prob

# UI 
st.title("AML Transaction Monitoring System")

tab1, tab2 = st.tabs(["🔍 Single Transaction Check", "📁 Bulk Transaction Upload"])

# Single
with tab1:
    with st.form("txn_form"):
        st.subheader("Transaction Information")
        tx_amount = st.number_input("Transaction Amount", min_value=0.0)
        timestamp = st.text_input("Timestamp (YYYY-MM-DD HH:MM:SS)", value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        st.subheader("Account IDs")
        sender_id = st.number_input("Sender Account ID", min_value=0)
        receiver_id = st.number_input("Receiver Account ID", min_value=0)

        st.subheader("Additional Information")
        sender_country_override = st.text_input("Sender Country")
        receiver_country_override = st.text_input("Receiver Country")
        sender_type_override = st.text_input("Sender Account Type")
        receiver_type_override = st.text_input("Receiver Account Type")

        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        try:
            hour = pd.to_datetime(timestamp).hour

            sender_info = get_account_info(sender_id, "SENDER")
            receiver_info = get_account_info(receiver_id, "RECEIVER")

            sender_country = sender_country_override or sender_info['SENDER_COUNTRY']
            receiver_country = receiver_country_override or receiver_info['RECEIVER_COUNTRY']
            sender_type = sender_type_override or sender_info['SENDER_ACCOUNT_TYPE']
            receiver_type = receiver_type_override or receiver_info['RECEIVER_ACCOUNT_TYPE']

            input_data = {
                "TX_AMOUNT": tx_amount,
                "HOUR": hour,
                "SENDER_COUNTRY": sender_country,
                "RECEIVER_COUNTRY": receiver_country,
                "SENDER_ACCOUNT_TYPE": sender_type,
                "RECEIVER_ACCOUNT_TYPE": receiver_type
            }

            prediction, prob = make_prediction(input_data)

            if prob > threshold:
                st.error(f"Fraud Likely (Probability: {prob:.2%})")
            else:
                st.success(f"Transaction Seems Legitimate (Probability: {prob:.2%})")

            sender_alerts = get_alerts(sender_id)
            receiver_alerts = get_alerts(receiver_id)

            if sender_alerts:
                st.warning("Alerts found for SENDER:")
                for alert in sender_alerts:
                    st.markdown(f"- **{alert['ALERT_TYPE']}**: {alert['ALERT_DESC']}")

            if receiver_alerts:
                st.warning(" Alerts found for RECEIVER:")
                for alert in receiver_alerts:
                    st.markdown(f"- **{alert['ALERT_TYPE']}**: {alert['ALERT_DESC']}")

        except Exception as e:
            st.error(f"Error: {e}")

# Bulk
with tab2:
    st.subheader("Upload a CSV file for Bulk Transaction Check")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data", df.head())

            df['HOUR'] = pd.to_datetime(df['TIMESTAMP']).dt.hour
            df['SENDER_COUNTRY'] = df['SENDER_ACCOUNT_ID'].apply(lambda x: get_account_info(x, "SENDER")['SENDER_COUNTRY'])
            df['RECEIVER_COUNTRY'] = df['RECEIVER_ACCOUNT_ID'].apply(lambda x: get_account_info(x, "RECEIVER")['RECEIVER_COUNTRY'])
            df['SENDER_ACCOUNT_TYPE'] = df['SENDER_ACCOUNT_ID'].apply(lambda x: get_account_info(x, "SENDER")['SENDER_ACCOUNT_TYPE'])
            df['RECEIVER_ACCOUNT_TYPE'] = df['RECEIVER_ACCOUNT_ID'].apply(lambda x: get_account_info(x, "RECEIVER")['RECEIVER_ACCOUNT_TYPE'])
            df['AMOUNT_BIN'] = pd.cut(df['TX_AMOUNT'], bins=amount_bins, labels=amount_labels)

            input_df = df[[
                'TX_AMOUNT', 'HOUR', 'SENDER_COUNTRY', 'RECEIVER_COUNTRY',
                'SENDER_ACCOUNT_TYPE', 'RECEIVER_ACCOUNT_TYPE', 'AMOUNT_BIN'
            ]]
            df['FRAUD_PROB'] = model.predict_proba(input_df)[:, 1]
            df['IS_FRAUD'] = df['FRAUD_PROB'] > threshold

            st.success("Predictions complete!")
            st.dataframe(df[['TX_AMOUNT', 'SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'FRAUD_PROB', 'IS_FRAUD']].sort_values(by="FRAUD_PROB", ascending=False))

        except Exception as e:
            st.error(f"Error processing file: {e}")
