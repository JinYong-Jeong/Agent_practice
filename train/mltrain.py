from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from dataready import df_preset
import pickle

# LIME 라이브러리 import
from lime.lime_tabular import LimeTabularExplainer

target = 'Survived'
x = df_preset.drop(target, axis=1)
y = df_preset[target]

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

# 모델 저장 (.pkl)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# K-Fold CV로 성능 검증
cv_score = cross_val_score(model, X_train, y_train, cv=10)
print('cv_score :', cv_score)
print('mean cv_score :', cv_score.mean())

# 예측
y_pred = model.predict(X_valid)

# 평가
print(classification_report(y_valid, y_pred))
print('Acc Score :', accuracy_score(y_valid, y_pred))
print('AUC Score :', roc_auc_score(y_valid, y_pred))

# LIME XAI 설명자 생성
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Not Survived', 'Survived'],
    mode='classification'
)

# X_valid의 첫 번째 샘플에 대해 XAI 결과 생성
idx = 0
exp = explainer.explain_instance(
    X_valid.values[idx],
    model.predict_proba,
    num_features=5
)

# API 응답 예시: 예측값 + XAI 결과
api_response = {
    "prediction": int(y_pred[idx]),  # 0 또는 1
    "xai_result": exp.as_list()      # [(feature, 영향도), ...]
}
print("API Response Example:", api_response)