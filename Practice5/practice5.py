# PythonML Practice 5: 다항 규제 회귀 기반 성적 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset

# [1] 데이터 분할: 범주형 변수를 제외한 6개 특성을 독립변수로, `exam_score`를 타깃으로 설정하고 8:2 비율 로 학습 및 검증 세트를 분리하시오.
# [2] 모델 전수 탐색: `LinearRegression`, `Ridge`, `Lasso` 모델과 다항 확장, 다양한 규제 강도 조합을 모두 학습시키시오.
# [3] 최적 모델 선정: 테스트 데이터셋(`X_test`) 기준 최고의 결정계수를 달성하는 최적의 알고리즘, 차수, 알파 값을 자동 도출하고 추론 엔진에 매핑하시오.
# [4] 추론 함수 구현: 새로운 학생의 6가지 특성 데이터를 인자로 받아 최적 모델의 다항 구조와 스케일링 기준을 거쳐 성적을 예측하는 함수를 구현하시오.
# [5] 샘플 데이터 검증: 구현된 함수에 두 가지 대조군 샘플을 대입하여 시험성적을 예측하시오.
    # study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
    # study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50
    
# [1] 
import pandas as pd
df = pd.read_csv('./Practice5/student_dataset_10000_rows.csv')
# 범주형(기준으로 나누어진 자료들) 수치형 (연속된 수)
# 독립변수(특성) : study_hours,attendance,sleep_hours,internet_usage,assignments_completed,previous_score,exam_score
# 종속변수(타깃) : exam_score
student_full = df[['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']]
student_target = df['exam_score'].values