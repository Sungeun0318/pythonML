# [1] 숭어 의 '길이','높이','두께' (3가지특성)  무게 (1가지타깃)
import pandas as pd 
df = pd.read_csv( './day02/Fish.csv')
perch = df[ df['Species'] == 'Perch' ]
perch_full = perch[ ['Length2' , 'Height', 'Width'] ]
perch_weight = perch[ 'Weight'].values

# [2] 훈련 세트 와 테스트 세트 분리 *모델검증 용도*
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split( perch_full , perch_weight , test_size=0.2 , random_state = 42  )

# [3] 특성 공학 , 다항 특성 제공 
# 다양한 특성을 추가로 만들어서 모델이 다양한 구조로 이해하기 위한 방법 ( *여러개 재료/자료 만들기* )
from sklearn.preprocessing import PolynomialFeatures
# 예제1] 1(기본값) , 2 , 2*2
poly = PolynomialFeatures() # 객체 생성 
poly.fit( [[ 2 ]] )
print( poly.transform( [[ 2 ]] ) )      # [[1. 2. 4.]]

# 예제2] 1(기본값제외) 2 , 3 , 2*2 , 3*3 , 2*3
poly = PolynomialFeatures( include_bias = False ) # 기본 편향 없음( 기본 1 제외 )
poly.fit( [[ 2, 3 ]] )
print( poly.transform( [[ 2 , 3 ]]))    # [[2. 3. 4. 6. 9.]]

# 적용 : '길이','높이','두께' 3가지의 특성이 갖는다.
poly = PolynomialFeatures( include_bias= False ) # 다항 특성 객체 생성 
poly.fit( train_input ) # 학습할 특성들을 대입한다.
train_poly = poly.transform( train_input )
test_poly = poly.transform( test_input ) 
print( train_poly.shape ) # 3가지 특성 --> 9개의 특성으로 변경된다.
# T2-02( 직접 제곱** ) , T2-03( PolynomialFeatures )

# [4] 다항 회귀 모델 학습 
from sklearn.linear_model import LinearRegression # 회귀 모델 
lr = LinearRegression()
lr.fit( train_poly , train_target )

# [5] 평가 , 1에 가까울수록 정확도가 크다.
print( lr.score( test_poly , test_target ) ) # 0.9764933250721679 # 테스트용은 훈련 이후에 평가 목적으로 사용됨.

# ============================================ # 
# 스코어(점수) == 회귀 모델 에서는 결정계수
# 계수란? 기울기 와 가중치 뜻 한다. # 즉] 어떠한 예측 결과에 얼마나 중요한 비중 차지 하는지 # 예측값 마다 존재!

# 결정계수란? 0 ~ 1 사이의 값으로 예측한 값이 얼마나 잘 설명 하는지 나타내는 수치 
# 결정계수 계산식 *왜? K-NN모델은 전체 계산식이 아닌 근접한 이웃 이용한 계산식 이므로 한계점 *
    # 타깃의 총 변동량 = SS_TOT = sum( (실제값 - 실제값평균)**2 )
    # 타깃의 잔차 변동량 = SS_RES = sum( (실제값 - 예측값 )**2 )
    # 1(100%) - ( ss_res/ss_tot )

# [6] 과대적합 확인 ( 5차 항 으로 표현 ) , degree=차수
poly = PolynomialFeatures( degree=5 , include_bias=False )
poly.fit( train_input )
train_poly = poly.transform( train_input )   
test_poly = poly.transform( test_input )         
print( train_poly.shape ) # 2차항에서는 3 -> 9  # 5창에서는 3 -> 55 
# 모델 학습 
lr.fit( train_poly , train_target ) # 55특성 으로 학습 
# 모델 평가 
# 과대적합 : 특정한 자료에만 과도한 학습을 통해 학습된것만 예측 하고 새로운 자료에 대해서는 예측 불가능 하다.
# 적절한 차수 선택한 모델의 최적화 조절한다.!
print( lr.score( train_poly , train_target )) # 0.999999999999193   # 훈련 데이터로는 100점 가깝다.
print( lr.score( test_poly , test_target  ) ) # -74.97494194987092  # 테스트 데이터로는 마이너스 이다.

# [7] 규제 하기 위한 전처리 ( 스케일링 )
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()  # 스케일링 객체 생성 
# 과대적합된 자료들을 스케일링 # 서로 다른 특성들 간에 (의미)단위가 다르므로 동일한 (의미)단위 사용하기 위해 ( 0 ~ 1 ) , 예] 몸무게 50~100 , 키 140~200
ss.fit( train_poly ) # 3개 특성이 55개 특성으로 과대적합된 상태의 표준편차/평균 계산 
train_scaled = ss.transform( train_poly ) 
test_scaled = ss.transform( test_poly )

# 릿지/라쏘 회귀들은 과적합된 자료들을 자동으로 특성제거/완만하게 해준다.
# [8] 릿지 회귀 : 가중치 줄여가면서 완전한 선 만들기 목적
# 알파( alpha = 규제단위 ) , alpha 단위가 크면 클수록 가중치(기울기) 0 으로 가깝다 만든다. # 규제강도
# 예] 길이 50 -> 10 줄였을때 성능이 줄면 중요한 계수 이다. 
# 예] 너비 50 -> 10 줄였을때 성능이 차이가 없으면 중요한 계수가 아니다.
from sklearn.linear_model import Ridge
ridge = Ridge() # 릿지 모델 객체 생성 
ridge.fit( train_scaled , train_target )    # 스케일링된 과대적합 자료 학습 

alpha_list = [ 0.001 , 0.01 , 0.1 , 1 , 10 , 100 ]
for alpth in alpha_list : 
    ridge = Ridge( alpha= alpth )   # 0.001 ~ 100 까지 반복 하면서 알파값 대입 
    ridge.fit( train_scaled , train_target ) # 서로 다른 알파값에 따른 학습 
    print( " --------------- 구분선 ----------- " , alpth )
    print( ridge.score( train_scaled , train_target ) ) # 평가1 
    print( ridge.score( test_scaled , test_target ) ) # 평가2
    # 최적: 학습평가 와 테스트평가 간의 격차가 적은것 : 알파10 가장 적합한 하이퍼파리미터 이다.

# [9] 라쏘 회귀 : 서로 특성 간의 관계없는 특성들을 (0)제거 하는 목적
# 알파( alpth = 규제강도 )
# 특정한 특성의 값을 변경했을 때 오차가 거의 없으면 관계 없다고 판단!
# 즉] 관계 없는 특성은 0으로 제거한다.
# 예] 길이 50 -> 30 줄였을 때 성능/오차 없다 < 필요없는 특성 > 0으로 변경
# 예] 너비 50 -> 30 줄였을 때 성능/오차 있다 < 중요한 특성 > 그대로 

from sklearn.linear_model import Lasso
lasso = Lasso( alpha= 10 ) # 라쏘 모델 객체 생성  # alpah = 규제강도
lasso.fit( train_scaled , train_target ) # 라쏘 모델 학습 

print( " --------------- 구분선 ----------- " , 10 )
print( lasso.score( train_scaled , train_target ) )
print( lasso.score( test_scaled , test_target ) )
