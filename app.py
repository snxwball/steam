# app.py (เวอร์ชันแก้ไข - แก้ปัญหาโมเดล bias)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Steam Game Predictor", page_icon="🎮", layout="centered")

# ฟังก์ชันเทรนโมเดล (แก้ไขแล้ว)
@st.cache_resource
def train_model():
    df = pd.read_csv('steam_games_dataset.csv')
    
    # Preprocessing
    bool_cols = ['windows', 'mac', 'linux', 'is_free']
    for col in bool_cols:
        df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
    
    features = ['windows', 'mac', 'linux', 'achievements', 'screenshots', 'movies']
    X = df[features].fillna(0)
    y = df['is_free']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ✅ แก้ไข: ใช้ class_weight='balanced' เพื่อปรับสมดุลข้อมูล
    # ✅ แก้ไข: ลด C เพื่อลด overfitting
    # ✅ แก้ไข: ใช้ kernel='linear' เหมาะกับข้อมูลน้อย
    model = SVC(
        kernel='linear',          # เปลี่ยนเป็น linear
        C=0.1,                    # ลด C จาก 1.0 เป็น 0.1
        class_weight='balanced',  # ปรับสมดุล class
        probability=True,         # เปิดใช้ predict_proba
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # คำนวณความแม่นยำ
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0.0
    
    return model, scaler, accuracy, X, y, features

# โหลดโมเดล
model, scaler, accuracy, X, y, features = train_model()

# --- UI ---
st.title("🎮 Steam Game Predictor (SVM)")
st.markdown("ระบบทำนายว่าเกมบน Steam จะเป็น **Free-to-Play** หรือ **เกมเสียเงิน**")
st.markdown("---")

# แสดงข้อมูลโมเดล
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("จำนวนเกมใน Dataset", len(X))
with col_info2:
    st.metric("เกม Free-to-Play", int(y.sum()))
with col_info3:
    st.metric("เกม Paid", int(len(y) - y.sum()))

st.markdown("---")

st.subheader("กรอกข้อมูลของเกม")
col1, col2 = st.columns(2)

with col1:
    windows = st.selectbox("รองรับ Windows?", [True, False])
    mac = st.selectbox("รองรับ Mac?", [True, False])
    linux = st.selectbox("รองรับ Linux?", [True, False])

with col2:
    achievements = st.number_input("จำนวน Achievements", min_value=0, value=0, step=1)
    screenshots = st.number_input("จำนวน Screenshots", min_value=0, value=0, step=1)
    movies = st.number_input("จำนวน Movies / Trailers", min_value=0, value=0, step=1)

if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'windows': [int(windows)],
        'mac': [int(mac)],
        'linux': [int(linux)],
        'achievements': [achievements],
        'screenshots': [screenshots],
        'movies': [movies]
    })
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    # ✅ แสดงความมั่นใจในการทำนาย
    probabilities = model.predict_proba(input_scaled)[0]
    prob_free = probabilities[1] * 100
    prob_paid = probabilities[0] * 100
    
    st.markdown("---")
    st.subheader("🎯 ผลการทำนาย")
    
    if prediction[0] == 1:
        st.success(f"🎉 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **Free-to-Play**")
    else:
        st.error(f"💰 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **เกมเสียเงิน (Paid)**")
    
    # ✅ แสดงความมั่นใจ
    st.markdown("###  ความมั่นใจในการทำนาย")
    
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric("โอกาสเป็น Free-to-Play", f"{prob_free:.1f}%")
    with col_prob2:
        st.metric("โอกาสเป็น Paid Game", f"{prob_paid:.1f}%")
    
    # ✅ แสดงข้อมูลที่ใช้ทำนาย
    st.markdown("###  ข้อมูลที่คุณกรอก:")
    st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)

# --- แสดงข้อมูลใน Dataset ---
with st.expander("📖 ดูข้อมูลเกมใน Dataset"):
    st.dataframe(df[['name', 'is_free', 'achievements', 'screenshots', 'movies']])