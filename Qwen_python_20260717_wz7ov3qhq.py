# app.py
import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Steam Game Predictor", page_icon="🎮", layout="centered")

# โหลดโมเดลและ Scaler ที่บันทึกไว้
@st.cache_resource
def load_models():
    model = joblib.load('svm_steam_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_models()

# --- สร้าง UI บน Streamlit ---
st.title("🎮 Steam Game Predictor (SVM)")
st.markdown("ระบบทำนายว่าเกมบน Steam จะเป็น **Free-to-Play** หรือ **เกมเสียเงิน** โดยใช้โมเดล Machine Learning (SVM)")
st.markdown("---")

# สร้างฟอร์มให้ผู้ใช้กรอกข้อมูล
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

# ปุ่มกดทำนาย
if st.button("🔮 ทำนายผล", type="primary", use_container_width=True):
    # เตรียมข้อมูลในรูปแบบ DataFrame ให้ตรงกับตอนเทรน
    input_data = pd.DataFrame({
        'windows': [int(windows)],
        'mac': [int(mac)],
        'linux': [int(linux)],
        'achievements': [achievements],
        'screenshots': [screenshots],
        'movies': [movies]
    })
    
    # ปรับสเกลข้อมูลด้วย Scaler เดิมที่ใช้ตอนเทรน
    input_scaled = scaler.transform(input_data)
    
    # ทำนายผล
    prediction = model.predict(input_scaled)
    
    # แสดงผลลัพธ์
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🎉 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **Free-to-Play** สูง!")
    else:
        st.error("💰 ผลลัพธ์: เกมนี้มีแนวโน้มเป็น **เกมเสียเงิน (Paid)** สูง!")