import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import threading
import plotly.graph_objects as go
import base64

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #F5F5F5; }
    .stButton>button {
        background-color: #FF4040;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #E63900;
    }
    .watch-subheader {
        color: #FF4040;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stSlider>div>div>div {
        background-color: #FF4040;
    }
    </style>
""", unsafe_allow_html=True)

# Loading models and scaler
@st.cache_resource
def load_models_and_scaler():
    try:
        class_model = pickle.load(open('best_class_model.pkl', 'rb'))
        reg_model = pickle.load(open('best_reg_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return class_model, reg_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model or scaler file not found: {e}. Please run health_prediction_model.py to generate these files.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        return None, None, None

# Predicting health for a single data point
def predict_health(data, class_model, reg_model, scaler):
    features = ['SpO2 (%)', 'Temperature (°C)', 'Pulse (bpm)', 'Heart Rate (bpm)', 
                'Blood Pressure Systolic (mmHg)', 'Blood Pressure Diastolic (mmHg)', 
                'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    data_df = pd.DataFrame([data], columns=features)
    try:
        data_scaled = scaler.transform(data_df)
        class_pred = class_model.predict(data_scaled)[0]
        class_prob = class_model.predict_proba(data_scaled)[0][1]
        days_pred = max(0, round(float(reg_model.predict(data_scaled)[0])))
        return class_pred, class_prob, days_pred
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

# HTTP server for ESP32-S3 predictions
class PredictionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                class_model, reg_model, scaler = load_models_and_scaler()
                if class_model is None:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Failed to load models or scaler"}).encode('utf-8'))
                    return
                
                data_list = data.get("data", [])
                if len(data_list) != 12:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid input: Expected 12 features"}).encode('utf-8'))
                    return
                
                class_pred, class_prob, days_pred = predict_health(data_list, class_model, reg_model, scaler)
                if class_pred is None:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Prediction failed"}).encode('utf-8'))
                    return
                
                response = {
                    "is_recovered": int(class_pred),
                    "recovery_prob": float(class_prob),
                    "days_to_recovery": int(days_pred)
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Prediction failed: {str(e)}"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_http_server():
    server_address = ('0.0.0.0', 8000)
    httpd = socketserver.TCPServer(server_address, PredictionHandler)
    st.write("Starting HTTP server on port 8000 for ESP32-S3 predictions...")
    httpd.serve_forever()

# Streamlit app
def main():
    # Sidebar navigation
    st.sidebar.title("Health Tracka")
    page = st.sidebar.radio("Navigate", ["Home", "Manual Prediction", "Batch Prediction", "About"])

    # Load models and scaler
    class_model, reg_model, scaler = load_models_and_scaler()
    if class_model is None:
        return

    if page == "Home":
        st.title("Welcome to Health Prediction System")
        st.markdown("""
            <div class="section-container">
            <p style='font-size: 18px;'>Monitor your health with real-time predictions using wearable sensor data. 
            Navigate to <b>Manual Prediction</b> for single predictions or <b>Batch Prediction</b> for multiple records.</p>
            </div>
        """, unsafe_allow_html=True)
        # Add a placeholder image (base64-encoded to avoid file dependency)
        st.image("https://via.placeholder.com/600x200.png?text=Health+Prediction", caption="Track your recovery with ease")

    elif page == "Manual Prediction":
        st.title("Real-Time Health Prediction")
        with st.container():
            st.markdown("""
                <div class="section-container">
                <div class="watch-subheader">Please Enter your watch for real-time prediction</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Instructions expander
            with st.expander("How to Input Sensor Data"):
                st.write("""
                    Enter sensor readings from your smartwatch or medical device. Ensure values are within realistic ranges:
                    - **SpO2**: 0–100%
                    - **Temperature**: 30–45°C
                    - **Pulse/Heart Rate**: 30–200 bpm
                    - **Blood Pressure**: Systolic 80–200 mmHg, Diastolic 50–150 mmHg
                    - **Accelerometer/Gyroscope**: -2 to 2 g or rad/s
                """)

            # Input form with sliders
            col1, col2 = st.columns(2)
            with col1:
                spo2 = st.slider("SpO2 (%)", 0.0, 100.0, 95.0, help="Oxygen saturation (0-100%)")
                temp = st.slider("Temperature (°C)", 30.0, 45.0, 36.5, help="Body temperature")
                pulse = st.slider("Pulse (bpm)", 30, 200, 80, help="Pulse rate")
                hr = st.slider("Heart Rate (bpm)", 30, 200, 80, help="Heart rate")
                bp_sys = st.slider("Blood Pressure Systolic (mmHg)", 80, 200, 120, help="Systolic BP")
                bp_dia = st.slider("Blood Pressure Diastolic (mmHg)", 50, 150, 80, help="Diastolic BP")
            
            with col2:
                accel_x = st.slider("Accel_X (g)", -2.0, 2.0, 0.0, help="Accelerometer X-axis")
                accel_y = st.slider("Accel_Y (g)", -2.0, 2.0, 0.0, help="Accelerometer Y-axis")
                accel_z = st.slider("Accel_Z (g)", -2.0, 2.0, 0.0, help="Accelerometer Z-axis")
                gyro_x = st.slider("Gyro_X (rad/s)", -2.0, 2.0, 0.0, help="Gyroscope X-axis")
                gyro_y = st.slider("Gyro_Y (rad/s)", -2.0, 2.0, 0.0, help="Gyroscope Y-axis")
                gyro_z = st.slider("Gyro_Z (rad/s)", -2.0, 2.0, 0.0, help="Gyroscope Z-axis")

            # Buttons for prediction and clearing inputs
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_btn = st.button("Predict")
            with col_btn2:
                clear_btn = st.button("Clear Inputs")

            if clear_btn:
                st.session_state.clear()
                st.rerun()

            if predict_btn:
                data = [spo2, temp, pulse, hr, bp_sys, bp_dia, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                class_pred, class_prob, days_pred = predict_health(data, class_model, reg_model, scaler)
                
                if class_pred is not None:
                    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                    st.subheader("Prediction Results")
                    if class_pred == 1:
                        st.success("The patient is predicted to be fully recovered! 🎉")
                    else:
                        st.warning(f"The patient is not yet recovered. Estimated days to recovery: {days_pred}")
                        recovery_date = datetime.now() + timedelta(days=days_pred)
                        st.write(f"**Expected recovery date**: {recovery_date.strftime('%Y-%m-%d')}")
                    st.write(f"**Recovery probability**: {class_prob:.2%}")

                    # Plotly gauge for recovery probability
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=class_prob * 100,
                        title={'text': "Recovery Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#FF4040"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightcoral"},
                                {'range': [50, 80], 'color': "gold"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Batch Prediction":
        st.title("Batch Health Prediction")
        with st.container():
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.subheader("Upload Sensor Data (Excel)")
            uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
            
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file)
                    st.write("**Uploaded Data Preview**:")
                    st.dataframe(df.head())
                    
                    required_columns = ['SpO2 (%)', 'Temperature (°C)', 'Pulse (bpm)', 'Heart Rate (bpm)', 
                                       'Blood Pressure Systolic (mmHg)', 'Blood Pressure Diastolic (mmHg)', 
                                       'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
                    if not all(col in df.columns for col in required_columns):
                        st.error(f"Excel file must contain columns: {', '.join(required_columns)}")
                        return
                    
                    predictions = []
                    progress_bar = st.progress(0)
                    for i, (_, row) in enumerate(df[required_columns].iterrows()):
                        class_pred, class_prob, days_pred = predict_health(row.values, class_model, reg_model, scaler)
                        if class_pred is not None:
                            predictions.append({
                                'Is_Recovered': 'Yes' if class_pred == 1 else 'No',
                                'Recovery_Probability': f"{class_prob:.2%}",
                                'Days_to_Recovery': days_pred,
                                'Expected_Recovery_Date': (datetime.now() + timedelta(days=days_pred)).strftime('%Y-%m-%d')
                            })
                        progress_bar.progress((i + 1) / len(df))
                    
                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        st.write("**Batch Prediction Results**:")
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Download button for results
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="health_predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error processing Excel file: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    elif page == "About":
        st.title("About the Health Prediction System")
        st.markdown("""
            <div class='section-container'>
            <p style='font-size: 18px;'>This system uses machine learning to predict health recovery based on wearable sensor data. 
            It integrates with a smartwatch (ESP32-S3) to provide real-time predictions and supports batch processing of sensor data via Excel files.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Real-time predictions using SpO2, temperature, heart rate, and motion data</li>
                <li>Batch predictions for large datasets</li>
                <li>Interactive visualizations and user-friendly interface</li>
                <li>Integration with wearable devices via HTTP</li>
            </ul>
            <p>Developed as part of a health monitoring project, leveraging XGBoost and Random Forest models trained on a dataset with a [4787 813] class distribution.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Start HTTP server in a separate thread
    threading.Thread(target=run_http_server, daemon=True).start()
    # Run Streamlit app
    st.write("Starting Streamlit app on port 8501...")
    main()