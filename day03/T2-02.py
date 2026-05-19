
# 모델 : 데이터(자료)을 학습하는 프로그램/라이브러리
    # K-NN 모델 : 가까운 이웃 기준의 예측
    # KNeighborsClassifier()    k최근접 이웃 분류
    # KNeighborsRegressor()     k최근접이웃 회귀
         # 하이퍼파라미터(K) : 이웃개수(K) 직접 설정하여 최적의 K찾기
         # 학습특성의 형태는 2차원배열만 가능, T1-01(zip활용), T1-02(column_stack활용), T2-01(reshape)

# 학습 : 데이터(자료)의 규칙 찾는 과정
# 예측 : 학습된 모델로 새로운 데이터(결과) 추론 과정
# 특성 : 학습에 입력되는 정보   # 물고기의 '길이', '무게'
# 타깃 : 학습에 정답되는 정보   # 물고기의 '종류'
# 표준화(스케일링) : 0 ~ 1 사이로 크기 맞춤
    # StandardScaler() # .transform()
# 과소적합           : 너무 단순한 경우 # 이웃이 너무 많아서 기준 애매 모호
# 과대적합/과적합      : 너무 암기된 경우 # 이웃이 너무 적어서 특정 이웃 학습
# -----------------------------------------------------------------------------

# [1] 숭어의 '길이', '무게' : '길이'(특성)에 따른 '무게'(타깃) 예측
import pandas as pd
df = pd.read_csv('./day03/Fish.csv')
fish_data = df[df['Species'].isin(['Perch'])]
perch_length = fish_data['Length2'].values
perch_weight = fish_data['Weight'].values
print(perch_length, perch_weight) # 확인

# [2] 훈련 세트와 테스트 세트 분리,
from sklearn.model_selection import train_test_split
# train_test_split(특성, 타깃, test_size=비율, random_state=분리기준난수)
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, test_size=0.2, random_state=42)

# [3] 학습 하기전에 사이킷런 모델들은 2차원 배열만 가능하다. # 1차원을 2차원으로 만들어줌
train_input = train_input.reshape(-1, 1) # reshape(행개수, 열개수) # -1 행은 자동으로 설정, 1열은 1개 설정
test_input = test_input.reshape(-1, 1)

# [4] k-최근법 이웃 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor # 회귀
knr = KNeighborsRegressor() # 모델 객체 생성
knr.fit(train_input, train_target) # 모델 학습
print(knr.score(test_input,test_target)) # 모델 평가 0.9932626838364674

# [5] 임의의 값으로 예측하기, 임의의 길이 50 넣어서 무게 예측하자.
print(knr.predict([[50]]))  # [1010.] # 임의의 물고기 길이 50일 때 무게 예측
print(knr.predict([[100]])) # [1010.] # 임의의 물고기 길이 100일 때 무게 예측

# 문제점 : k-최근법 이웃의 문제점은 단순한 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 값으로 예측한다.
# 즉] 소규모 또는 간단한 예측 프로그램에서만 사용된다. 




# [1] 다른 모델 사용하기
from sklearn.linear_model import LinearRegression # 선형회귀 모델
lr = LinearRegression() # 모델 객체 생성
lr.fit(train_input, train_target) # 모델 학습
print(lr.score(test_input, test_target)) # 모델 평가
print(lr.predict([[50]])) # 모델 예측 [1238.3175398]
print(lr.predict([[100]])) # [3191.00026354]
# 직선 공식(1차 방정식) : y(예측) = w(가중치) x (특성) + b(절편)
# 즉] (물고기)무게 = 가중치 * (물고기)길이 + 절편
print(lr.coef_) # 기울기값 반환 [39.05365447]
    # 기울기(가중치) 공식 : x와 y의 편차 곱의 합 / x의 편차 제곱합
print(lr.intercept_) # y절편 반환 -714.3651839448925 # 편향 # x(물고기길이)가 0일 때 y의 값

# x와 y가 직선 관계이며 실 자료들은 물고기가 길이 1씩 증가할 때 무게가 꼭 비례 증가하지 않는다. <애매하다>
# 즉] 초반에는 길이에 따라 무게가 3배 증가 하다가 중/후반에는 무게가 2/1배 증가 할수 있다. # 사람 키(어릴 때 키각 자라고 나이들면 어느정도 고정)



# [2] 시각화
import matplotlib.pyplot as plt
plt.scatter(train_input, train_target) # x축 : 길이 y축 : 무게
plt.scatter(50, 1238) # 예측된 결괴 # 길이가 50일 때 무게는 1238일 것이다.
plt.scatter(100, 3191) # 예측된 결과 # 길이가 100일때 무게는 3191일 것이다
plt.plot([15, 100], lr.predict([[15], [100]])) # 회귀선 그리기 # 0 시작점, 100끝점
plt.show()
print(lr.score(test_input,test_target)) # 0.8359630155975615

# [3] (다항 : 여러개 항) 회귀
# x(특성) : 물고기 '길이'
# x(특성) 제곱 : 물고기 '길이'에 제곱 *최적의 제곱수 찾아서 정확도 최적화 한다.
# 가중치 : 기울기
# 절편 : y절편 / 편향
# 즉] x제곱 항목이 추가되면서 그래프가 U 또는 곡선 모양으로 나온다. 길이가 커질수록 무게는 뻥튀기되는 효과
import numpy as np
train_poly = np.column_stack((train_input ** 2, train_input)) # +더하기, ** 제곱 # [길이제곱, 길이]
print(train_poly)

lr = LinearRegression()
lr.fit(train_poly, train_target) # 다항으로 학습

# 예측할 자료, 길이 : 50인 무게 예측
print(lr.predict([[50**2, 50]])) # [1579.0440311]

# 여러개 에측
point = np.arange(15, 50) # 15부터 50까지(예측하고 싶은 범위) 1씩 증가 하는 리스트 반환
point_poly = np.column_stack((point ** 2, point)) # 15~50 제곱한 열, 15~50 열
print(point_poly)

# 시각화
plt.scatter(train_input, train_target) # 특성 자료
plt.plot(point, lr.predict(point_poly)) # 여러개 예측 시각화
plt.show()

test_poly = np.column_stack((test_input**2, test_input))
print(lr.score(test_poly, test_target)) # 다항 회귀 평가  # 0.9801885585527478

