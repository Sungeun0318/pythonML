
# [1] 여러가지 특성에 분류
import pandas as pd
df = pd.read_csv('./day05/Fish.csv')

# 어종 7개, Species
fish_target = df['Species']

# 특성 6개, Weight,Length1,Length2,Length3,Height,Width
fish_input = df[['Weight','Length1','Length2','Length3','Height','Width']]

# 훈련 / 테스트 분리
from sklearn.model_selection import train_test_split
train_input, train_target, test_input, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# [2] 이진 분류
