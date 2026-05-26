
# alcohol,sugar,pH,class
# [1]
import pandas as pd
df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol', 'sugar', 'pH']]  # 와인들의 속성 3개
target = df['class']                            # 1 : 화이트와인 0 : 레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)

# [2] 결정 트리(분류 모델)
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류
dt = DecisionTreeClassifier() # 모델 객체 생성
dt.fit(train_input, train_target) # 모델 학습
print(dt.score(train_input, train_target)) # 모델 정확도 # 0.9973316912972086
print(dt.score(test_input, test_target)) # 0.8492307692307692

# 모델 예측
print(dt.predict(test_input[: 5])) # 모델 예측(5개만) # [1. 0. 1. 1. 1.]

# [3] 결정 트리 시각화 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# max_depth = 가지수, 트리의 최대 깊이, 최대적합 억제 # 가지수가 너무 많아지면 과대적합 발생
# filled=True
plot_tree(dt, max_depth=3, feature_names = ['alcohol', 'sugar', 'pH'], class_names=['Red Wine', 'White Wine'], filled=True) # plot_tree(트리모델, max_depth = 뿌리 개수)
plt.show()
# 트리 : 전체적인 구조 그자체,
# 노드 : 사각형 상자 하나하나 의미, 가장 위에 있는 노드를 루트(root) 노트
# 노드 속성 : 
    # value = [예측타겟수] # [85, 2097] 0으로 예측하는 수가 85개, 1로 예측하는 수가 2097개 뜻
    # gini = 불순도 # 0.075 # 0으로 가까울수록 정확/순수하다
        # 0으로 가까울수록 순수(특정 예측값으로 모여)하다
        # 0.5에 가까울수록 혼란(예측값이 섞여)하다.
        # sugar =  특성 # sugar <= 0.45 보다 작으면 true(왼쪽노드로 이동), false(오른쪽노드로 이동)
        
# [4] 특성 중요도
# 각 특성이 트리 모델에 얼마나 중요한 역할 하는지 수치 # 합은 1
print(dt.feature_importances_) # [0.23656455 0.51860098 0.24483448]
print(dt.feature_importances_[0]) # alcohol # sugar # pH

# [5] 최소한의 불순도(gini) 설정, 최적의 파라미터
dt = DecisionTreeClassifier(random_state = 42, min_impurity_decrease = 0.0005)
dt.fit(train_input, train_target) 
print(dt.score(train_input, train_target)) # 0.8975779967159278
print(dt.score(test_input, test_target)) # 0.8590769230769231 # 과대적합 최소화
    