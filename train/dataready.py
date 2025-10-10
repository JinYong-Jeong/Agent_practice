
import sklearn as sk
import pandas as pd

path = 'https://raw.githubusercontent.com/khw11044/csv_dataset/master/titanic.0.csv'
df = pd.read_csv(path)

drop_cols = ['PassengerId', 'Age', 'SibSp']
df_del = df.drop(drop_cols, axis=1)

df_na = df_del.dropna(subset=['Embarked'], axis=0)  # 행은 axis=0
df_na = df_na.drop('Cabin', axis=1)  # 행은 axis=0

selected = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']
df_sel = df_na[selected]

# 가변수화 대상: Pclass, Sex, Embarked
dumm_cols = ['Pclass', 'Sex', 'Embarked']

# 가변수화
df_preset = pd.get_dummies(df_sel, columns=dumm_cols, drop_first=True, dtype=int)

# print(df_preset.head())