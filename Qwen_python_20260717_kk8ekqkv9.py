# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

print("🔄 กำลังโหลดข้อมูล...")
df = pd.read_csv('steam_games_dataset.csv')

# 1. Data Preprocessing (เตรียมข้อมูล)
# แปลงคอลัมน์ Boolean ให้เป็น 1/0 และจัดการกับค่า NaN
bool_cols = ['windows', 'mac', 'linux', 'is_free']
for col in bool_cols:
    df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

# เลือกฟีเจอร์ (Features) ที่จะใช้ทำนาย
features = ['windows', 'mac', 'linux', 'achievements', 'screenshots', 'movies']
X = df[features].copy()
y = df['is_free'].copy() # Target: 1 = Free, 0 = Paid

# เติมค่า NaN ในฟีเจอร์ด้วย 0
X = X.fillna(0)

# 2. Train-Test Split (แบ่งข้อมูลสำหรับเทรนและทดสอบ)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (ปรับสเกลข้อมูล)
# SVM อ่อนไหวต่อสเกลของข้อมูลมาก จึงจำเป็นต้องใช้ StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (เทรนโมเดล SVM)
print("🤖 กำลังเทรนโมเดล SVM...")
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 5. Model Evaluation (ประเมินผลโมเดล)
y_pred = svm_model.predict(X_test_scaled)
print("\n✅ ผลการประเมินโมเดล:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Paid (0)', 'Free (1)']))

# 6. Save Model (บันทึกโมเดลและ Scaler ลงไฟล์)
joblib.dump(svm_model, 'svm_steam_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 บันทึกโมเดล (svm_steam_model.pkl) และ Scaler (scaler.pkl) เรียบร้อยแล้ว!")