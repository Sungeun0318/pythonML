
# [1]
import pandas as pd
df = pd.read_csv('./day06/wine.csv')
data = df[['alcohol', 'sugar', 'pH']]  # 와인들의 속성 3개
target = df['class']                            # 1 : 화이트와인 0 : 레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)

# 트리의 앙상블 : 학습한 모델에서 오답들을 서로 상쇄하고 정답을 강화하여 예측정확도 높여 과대적합 방지하는 방법 # 여러가지 방법 존재
# [2] 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(oob_score=True, n_jobs= -1, random_state=42)

# 교차 검증
from sklearn.model_selection import cross_validate
scores = cross_validate(rf, train_input, train_target, n_jobs=-1)
print(scores) # 'test_score': array([0.88      , 0.89948718, 0.90349076, 0.89117043, 0.88501027])
import numpy as np
print(np.mean(scores['test_score'])) # 0.8918317274785446
