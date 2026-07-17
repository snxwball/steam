import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับตกแต่ง
st.markdown("""
<style>
    /* ปรับแต่งพื้นหลังหลัก */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
    }
    
    /* การ์ดสีขาวสำหรับฟอร์ม */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* ปรับแต่ง header */
    h1 {
        color: #2d3748;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* ปรับแต่ง subheader */
    .subtitle {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* ปรับแต่ง input fields */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 12px;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    /* ปรับแต่งปุ่ม */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* การ์ดสำหรับแสดงผล */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
        max-width: 600px;
    }
    
    .success-box {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* ปรับแต่ง sidebar */
    .sidebar-content {
        background: white;
        padding: 2rem;
        border-radius: 15px;
    }
    
    /* Label styling */
    .stNumberInput label, .stSelectbox label {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    /* Container styling */
    .css-1r6slb0 {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# ส่วนหัวของเว็บ
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>ระบบพยากรณ์ความเสี่ยงโรคหัวใจด้วย Machine Learning</p>", unsafe_allow_html=True)

# สร้างฟอร์มใน container สวยๆ
with st.container():
    # แบ่งเป็น 2 คอลัมน์
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 ข้อมูลส่วนตัว")
        age = st.number_input("Age (อายุ)", min_value=20, max_value=100, value=56, step=1)
        sex = st.selectbox("Sex (เพศ)", options=[0, 1], format_func=lambda x: "👨 Male" if x==1 else "👩 Female")
        chest_pain = st.selectbox("Chest Pain Type (ประเภทอาการเจ็บหน้าอก)", 
                                  options=[1, 2, 3, 4],
                                  format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                                                        3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
        resting_bp = st.number_input("Resting BP (ความดันโลหิต)", min_value=80, max_value=200, value=124, step=1)
        cholesterol = st.number_input("Cholesterol (คอเลสเตอรอล)", min_value=100, max_value=600, value=200, step=1)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", 
                                  options=[0, 1],
                                  format_func=lambda x: "Yes" if x==1 else "No")
    
    with col2:
        st.markdown("### 📊 ผลการตรวจ")
        resting_ecg = st.selectbox("Resting ECG", 
                                   options=[0, 1, 2],
                                   format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LVH"}[x])
        max_hr = st.number_input("Max Heart Rate (อัตราการเต้นหัวใจสูงสุด)", min_value=50, max_value=220, value=147, step=1)
        exercise_angina = st.selectbox("Exercise Angina (เจ็บหน้าอกเมื่อออกกำลังกาย)", 
                                       options=[0, 1],
                                       format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.02, step=0.01)
        st_slope = st.selectbox("ST Slope", 
                                options=[1, 2, 3],
                                format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # ปุ่มทำนาย
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 ทำนายผลความเสี่ยง", use_container_width=True)
    
    # แสดงผลการทำนาย
    if predict_button:
        try:
            model = load_model()
            
            # เตรียมข้อมูล
            input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, 
                                    fasting_bs, resting_ecg, max_hr, exercise_angina, 
                                    oldpeak, st_slope]])
            
            # ทำนาย
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            
            # แสดงผล
            if prediction == 1:
                st.markdown("""
                    <div class='result-card'>
                        <div class='danger-box'>
                            <h2 style='color: white; margin: 0;'>️ ผลการพยากรณ์</h2>
                            <h3 style='color: white; margin: 10px 0;'>มีความเสี่ยงเป็นโรคหัวใจ</h3>
                            <p style='font-size: 1.2rem; margin: 0;'>ความน่าจะเป็น: {:.2%}</p>
                        </div>
                        <p style='color: #4a5568; margin-top: 1rem;'>
                            <strong>คำแนะนำ:</strong> ควรปรึกษาแพทย์เพื่อการวินิจฉัยและรักษาอย่างถูกต้อง
                        </p>
                    </div>
                """.format(probability[1]), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='result-card'>
                        <div class='success-box'>
                            <h2 style='color: white; margin: 0;'>✅ ผลการพยากรณ์</h2>
                            <h3 style='color: white; margin: 10px 0;'>ไม่มีความเสี่ยงเป็นโรคหัวใจ</h3>
                            <p style='font-size: 1.2rem; margin: 0;'>ความน่าจะเป็น: {:.2%}</p>
                        </div>
                        <p style='color: #4a5568; margin-top: 1rem;'>
                            <strong>คำแนะนำ:</strong> รักษาสุขภาพให้ดีอย่างต่อเนื่อง
                        </p>
                    </div>
                """.format(probability[0]), unsafe_allow_html=True)
            
            # แสดงกราฟความน่าจะเป็น
            st.markdown("### 📈 ความน่าจะเป็น")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="ความเสี่ยงโรคหัวใจ", value=f"{probability[1]:.2%}")
            with col2:
                st.metric(label="ไม่มีความเสี่ยง", value=f"{probability[0]:.2%}")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")

# Sidebar สำหรับข้อมูลเพิ่มเติม
with st.sidebar:
    st.markdown("###  เกี่ยวกับเรา")
    st.info("ระบบนี้ใช้ Machine Learning Model ในการพยากรณ์ความเสี่ยงโรคหัวใจ")
    
    st.markdown("### 📖 คำอธิบายตัวแปร")
    st.markdown("""
    - **Age**: อายุ (ปี)
    - **Sex**: เพศ (ชาย/หญิง)
    - **Chest Pain Type**: ประเภทอาการเจ็บหน้าอก
    - **Resting BP**: ความดันโลหิตขณะพัก
    - **Cholesterol**: ระดับคอเลสเตอรอล
    - **Fasting BS**: ระดับน้ำตาลในเลือด
    - **Resting ECG**: ผลการตรวจคลื่นหัวใจ
    - **Max HR**: อัตราการเต้นหัวใจสูงสุด
    - **Exercise Angina**: อาการเจ็บหน้าอกเมื่อออกกำลังกาย
    - **Oldpeak**: ST depression
    - **ST Slope**: ความชันของ ST segment
    """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #718096;'>Developed with ❤️ using Streamlit</p>", unsafe_allow_html=True)