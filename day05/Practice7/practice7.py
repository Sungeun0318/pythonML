# PythonML Practice7: 로지스틱 분류
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 독립/종속 변수 추출
# 파일명: ./Iris.csv
# 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' 4개 열을 독립 변수 X로,
# 'Species' 열을 종속 변수 y로 추출하세요.
import pandas as pd
df = pd.read_csv('./day05/Practice7/Iris.csv')

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# [단계 2] 훈련용 / 테스트용 데이터 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(x, y, test_size=0.3, random_state=42)


# [단계 3] 데이터 표준화 (Standardization), 스케일러
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# [단계 4] 로지스틱 분류 모델 학습 (Logistic Regression)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.predict(test_scaled[: 3]))
print(lr.predict_proba(test_scaled[ : 3]))


# [단계 5] 모델 평가 및 분류 정확도(Accuracy) 확인 * 테스트 세트의 정확도가 0.95 이상이 나오도록 설정
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# [단계 6] 학습한 종속 변수 출력
print(lr.classes_) # ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

# [단계 7] 테스트 세트의 앞선 5개 샘플 데이터에 대해 모델이 예측한 클래스를 출력하세요.

import numpy as np
a = lr.predict_proba(test_scaled[ : 5 ])
print(np.round(a, decimals=3))
# [[0.    0.989 0.01 ]
#  [0.995 0.005 0.   ]
#  [0.    0.    1.   ]
#  [0.    0.954 0.046]
#  [0.    0.957 0.043]]
