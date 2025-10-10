import pickle
import pandas as pd

# input_data = {
#     'Pclass': 3,
#     'Sex': 'male',
#     'Age': 22,
#     'SibSp': 1,
#     'Parch': 0,
#     'Fare': 7.25,
#     'Embarked': 'S'
# }
# 없는 컬럼은 0으로 추가
pclass = int(input("Pclass(1, 2, 3): "))
sex = input("Sex(male/female): ")
fare = float(input("티켓 가격(예: 7.25): "))
embarked = input("탑승구(C:셰르부르/Q:퀸스타운/S:사우스햄프턴): ")

input_data = {
    'Pclass': pclass,
    'Sex': sex,
    'Fare': fare,
    'Embarked': embarked
}

input_df = pd.DataFrame([input_data])

# 실제 학습에 사용된 컬럼 리스트
feature_columns = ['Fare', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Embarked_Q', 'Embarked_S']

# 입력값을 원핫인코딩 (Pclass, Sex, Embarked)
input_df_encoded = pd.get_dummies(input_df, columns=['Pclass', 'Sex', 'Embarked'])

# 없는 컬럼은 0으로 추가
for col in feature_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# 컬럼 순서 맞추기
input_df_encoded = input_df_encoded[feature_columns]

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ...existing code...
print(f"입력값: Pclass={pclass}, Sex={sex}, Fare={fare}, Embarked={embarked}")

# 변환된 값 출력
print(f"Sex_male: {input_df_encoded['Sex_male'].iloc[0]}, Embarked_Q: {input_df_encoded['Embarked_Q'].iloc[0]}, Embarked_S: {input_df_encoded['Embarked_S'].iloc[0]}")

prediction = model.predict(input_df_encoded)
print("예측 결과:", int(prediction[0]))