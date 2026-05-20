# Practice1: Iris 데이터셋을 활용한 꽃 품종 분류
# https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 확인
# 파일명: ./Iris.csv
import pandas as pd
df = pd.read_csv('./Practice/Iris.csv')
print(df.head())

# [단계 2] 특정 품종 추출 (데이터 필터링)
# 전체 데이터 중 Species가 'Iris-setosa'인 데이터와 'Iris-versicolor'인 데이터만 각각 추출하여 별도의 데이터프레임(setosa_df, versicolor_df)에 저장하세요.
setosa_df = df[df['Species'] == 'Iris-setosa']
versicolor_df = df[df['Species'] == 'Iris-versicolor']

# [단계 3] 특성 데이터 추출 (리스트 변환)
# 각 품종의 꽃잎 길이(PetalLengthCm)와 꽃잎 너비(PetalWidthCm)를 추출하여 파이썬 리스트로 변환하세요.
setosa_length = setosa_df['PetalLengthCm'].tolist()
setosa_width = setosa_df['PetalWidthCm'].tolist()
versicolor_length = versicolor_df['PetalLengthCm'].tolist()
versicolor_width = versicolor_df['PetalWidthCm'].tolist()
print(setosa_length, setosa_width)
print(versicolor_length, versicolor_width)

# [단계 4] 데이터 시각화
# 추출한 두 품종의 데이터를 산점도(Scatter plot)로 그리세요.
# X축: 꽃잎 길이(length), Y축: 꽃잎 너비(width)
# 두 품종의 색상을 다르게 표시하고 축 이름을 설정하세요.
import matplotlib.pyplot as plt

plt.scatter(setosa_length, setosa_width, label = 'Setosa')
plt.scatter(versicolor_length, versicolor_width, label = 'Versicolor')

plt.xlabel('length')
plt.ylabel('width')
plt.show()

# [단계 5] 학습 데이터 구성 (2차원 리스트)
# 두 품종의 길이를 합친 전체 길이 리스트와 너비를 합친 전체 너비 리스트를 만듭니다.
# zip() 함수와 리스트 내포를 사용하여 [길이, 너비] 형태의 2차원 리스트 iris_data를 생성하세요.
length = setosa_length + versicolor_length
width = setosa_width + versicolor_width
iris_data = [[l, w] for l, w in zip(length, width)]
print(iris_data)
print(setosa_df.info)
print(versicolor_df.info)


# [단계 6] 정답 데이터(Target) 생성
# 각 데이터에 대한 정답 리스트 iris_target을 생성하세요.
# Iris-setosa: 1 
# Iris-versicolor: 0
iris_target = [1] * 50 + [0] * 50
print(iris_target)


# [단계 7] 모델 학습 및 평가
# KNeighborsClassifier 모델 객체를 생성하세요.
# iris_data와 iris_target을 사용하여 모델을 학습(fit)시키세요.
# 학습된 모델의 정확도(score)를 출력하세요.

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(iris_data, iris_target)
print(kn.score(iris_data, iris_target))


# [단계 8] 예측 및 시각화
# 임의의 데이터 [꽃잎 길이 2.0, 꽃잎 너비 0.5]를 가진 꽃은 어떤 품종으로 예측되는지 코드를 작성하세요.
print(kn.predict([[2.0, 5.0]]))

plt.scatter(setosa_length, setosa_width)
plt.scatter(versicolor_length, versicolor_width)
plt.scatter(2.0, 0.5)
plt.show()
# versicolor 예측됨
