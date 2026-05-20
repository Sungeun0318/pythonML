# PythonML Practice4: 다항 회귀를 이용한 비선형 데이터 예측
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 특정 품종 추출
# 파일명: ./Iris.csv
# 전체 데이터 중 Species가 'Iris-setosa'인 데이터만 필터링하여
# 'SepalLengthCm' 열을 iris_length로, 'SepalWidthCm' 열을 iris_width로 추출하세요.
import pandas as pd
df = pd.read_csv('./Practice4/Iris.csv')
print(df.head())

setosa_df = df[df['Species'] == 'Iris-setosa']

iris_length = setosa_df['SepalLengthCm'].values
iris_width = setosa_df['SepalWidthCm'].values

# [단계 2] 훈련용 / 테스트용 데이터 분리
# train_test_split() 함수를 사용하여 학습용 데이터와 테스트용 데이터를 분리하세요.
# test_size는 0.3, random_state는 42로 설정하세요.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(iris_length, iris_width, test_size=0.3, random_state=42)

# [단계 3] 데이터 차원 변환 (Reshape)
# 사이킷런 모델 학습을 위해 1차원 배열인 train_input과 test_input을 [개수, 1] 형태의 2차원 배열로 변환하세요.
train_input =  train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# [단계 4] 단순 선형 회귀(Linear Regression) 모델 학습 및 평가
# 1) LinearRegression 객체를 생성하고 변환된 train_input 데이터로 모델을 학습하세요.
# 2) 학습된 모델의 훈련 세트와 테스트 세트의 결정계수(R^2)를 각각 출력하세요.
# 3) 모델의 기울기(coef_)와 절편(intercept_)을 각각 출력하세요.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.score(train_input, train_target)) # 0.5395298831723283
print(lr.score(test_input, test_target)) # 0.5423687498422732
print(lr.coef_) # [0.76182638]
print(lr.intercept_) # -0.38763106559375604


# [단계 5] 다항 회귀를 위한 데이터 전처리 (특성 추가)
# 꽃받침 너비의 비선형적 흐름을 반영하기 위해 numpy의 column_stack을 사용하여
# [길이^2, 길이] 형태의 2차원 배열 구조를 가진 train_poly와 test_poly를 각각 생성하세요.
import numpy as np
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))


# [단계 6] 다항 회귀 모델 학습 및 결정계수(R^2) 확인
# 1) LinearRegression 객체를 새로 생성하고 train_poly 데이터로 모델을 학습하세요.
# 2) 학습된 다항 회귀 모델의 훈련 세트와 테스트 세트의 결정계수(R^2)를 각각 출력하여 단순 선형 회귀와 비교하세요.
lr_poly = LinearRegression()
lr_poly.fit(train_poly, train_target)

print(lr_poly.score(train_poly, train_target)) # 0.5444662337785274
print(lr_poly.score(test_poly, test_target)) # 0.5413125141187336


# [단계 7] 다항 회귀 모델을 통한 임의의 값 예측
# 학습된 다항 회귀 모델을 사용하여 꽃받침 길이가 4.0일 때와 6.0일 때의 꽃받침 너비를 각각 예측하여 출력하세요.
print(lr_poly.predict([[4.0 ** 2, 4.0]])) # [2.83968176]
print(lr_poly.predict([[6.0 ** 2, 6.0]])) # [4.31020745]


# [단계 8] 다항 회귀 곡선 시각화
# 1) 4.0부터 6.0까지 0.1 간격으로 증가하는 배열(point)을 생성하고, 이를 다항 특성(point_poly)으로 변환하세요.
# 2) 산점도(scatter)를 이용해 원래의 train_input과 test_input 데이터를 시각화하세요.
# 3) 앞서 생성한 point와 다항 회귀 모델의 예측값(predict)을 사용하여 2차 방정식 형태의 곡선(회귀선)을 그리세요.
import matplotlib.pyplot as plt

point = np.arange(4.0, 6.1, 0.1)
point_poly = np.column_stack((point ** 2, point))

plt.scatter(train_input, train_target)
plt.scatter(test_input, test_target)

plt.plot(point, lr_poly.predict(point_poly))
plt.show()

# [단계 9] 단순 선형 회귀(1차 방정식)와 비교했을 때, 특성을 제곱한 다항 회귀(2차 방정식)가 가지는 수학적/표현적 장점을 주석으로 서술하시오.

# 직선이 아닌 곡선 형태로 데이터를 표현할 수 있어 비선형적인 데이터의 흐름이 더 잘 반영됨.