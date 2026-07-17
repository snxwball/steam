import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Steam Game Predictor", page_icon="", layout="centered")

# ฟังก์ชันเทรนโมเดล (ใช้ cache เพื่อไม่ต้องเทรนใหม่ทุกครั้ง)
@st.cache_resource
def train_model():
    # โหลดข้อมูล
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
    
    # Train SVM - ปรับปรุง parameters
    model = SVC(
        kernel='linear',
        C=0.1,
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # คำนวณความแม่นยำ
    y_pred = model.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred)
    
    return model, scaler, accuracy

# โหลดโมเดล
model, scaler, accuracy = train_model()

# --- UI ---
st.title("🎮 Steam Game Predictor (SVM)")
st.markdown("ระบบทำนายว่าเกมบน Steam จะเป็น **Free-to-Play** หรือ **เกมเสียเงิน**")
st.markdown("---")

# แสดงข้อมูลโมเดล
col_info1, col_info2 = st.columns(2)
with col_info1:
    st.metric("Model Accuracy", f"{accuracy:.2%}")
with col_info2:
    st.metric("Algorithm", "SVM (Linear)")

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
    
    # แสดงความมั่นใจในการทำนาย
    probabilities = model.predict_proba(input_scaled)[0]
    prob_free = probabilities[1] * 100
    prob_paid = probabilities[0] * 100
    
    st.markdown("---")
    st.subheader("🎯 ผลการทำนาย")
    
    # ✅ แก้ไขแล้ว - f-string ครบถ้วน
    if prediction[0] == 1:
        st.success("🎉 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **Free-to-Play** สูง!")
    else:
        st.error("💰 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **เกมเสียเงิน (Paid)** สูง!")
    
    # แสดงความมั่นใจ
    st.markdown("### 💡 ความมั่นใจในการทำนาย")
    
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric("โอกาสเป็น Free-to-Play", f"{prob_free:.1f}%")
    with col_prob2:
        st.metric("โอกาสเป็น Paid Game", f"{prob_paid:.1f}%")
    
    # แสดงข้อมูลที่ใช้ทำนาย
    st.markdown("### 📋 ข้อมูลที่คุณกรอก:")
    st.dataframe(input_data.T.rename(columns={0: 'ค่า'}), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>สร้างด้วย ❤️ โดย Streamlit + Scikit-learn</p>
    <p>Dataset: Steam Games Dataset</p>
</div>
""", unsafe_allow_html=True)