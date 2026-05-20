import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class CarService:
    def __init__(self):
        # 최적의 모델과 변환 도구들을 저장할 멤버 변수
        self.best_model = None
        self.best_poly = None
        self.best_scaler = None

    def train(self, carList):
        # 1. 데이터를 판다스 데이터프레임으로 변환
        df = pd.DataFrame(carList)
        
        # 2. 독립변수(X)와 종속변수(y) 분리
        # 자바 Entity의 필드명과 일치해야 함
        X = df[['평균연비', '누적주행거리키로', '출고후경과월수', '사고감가건수', '소유자변경횟수']]
        y = df['매매가격만원'].values
        
        # 3. 훈련 세트와 테스트 세트 분리 (8:2)
        train_input, test_input, train_target, test_target = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 4. 모델 전수 탐색 (Practice 5 패턴)
        optimization = []
        
        for degree in [1, 2, 3]:
            # 다항 특성 공학 적용
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            train_poly = poly.fit_transform(train_input)
            test_poly = poly.transform(test_input)
            
            # [A] 선형 회귀 학습 및 평가
            lr = LinearRegression()
            lr.fit(train_poly, train_target)
            r2 = lr.score(test_poly, test_target)
            optimization.append({'r2': r2, 'model': lr, 'poly': poly, 'scaler': None})
            
            # [B] 스케일링 적용 (릿지/라쏘용)
            ss = StandardScaler()
            train_scaled = ss.fit_transform(train_poly)
            test_scaled = ss.transform(test_poly)
            
            # 규제 강도(alpha)별 반복 탐색
            for alpha in [0.01, 0.1, 1, 10, 100]:
                # 릿지(Ridge) 모델
                ridge = Ridge(alpha=alpha)
                ridge.fit(train_scaled, train_target)
                r2 = ridge.score(test_scaled, test_target)
                optimization.append({'r2': r2, 'model': ridge, 'poly': poly, 'scaler': ss})
                
                # 라쏘(Lasso) 모델
                lasso = Lasso(alpha=alpha)
                lasso.fit(train_scaled, train_target)
                r2 = lasso.score(test_scaled, test_target)
                optimization.append({'r2': r2, 'model': lasso, 'poly': poly, 'scaler': ss})

        # 5. 결과 리스트에서 결정계수(r2)가 가장 큰 모델 찾기
        best_optimization = max(optimization, key=lambda x: x['r2'])
        
        # 최적 모델 및 도구 전역 저장
        self.best_model = best_optimization['model']
        self.best_poly = best_optimization['poly']
        self.best_scaler = best_optimization['scaler']
        
        print(f"학습 완료! 최적 R2: {best_optimization['r2']}")
        return {"status": "success", "r2": best_optimization['r2']}

    def predict(self, car):
        # 학습된 모델이 없으면 에러 반환
        if self.best_model is None:
            return {"status": "error", "message": "모델이 학습되지 않았습니다."}
        
        # 1. 입력 데이터를 2차원 리스트로 변환
        list_data = [[
            car.get('평균연비'), 
            car.get('누적주행거리키로'), 
            car.get('출고후경과월수'), 
            car.get('사고감가건수'), 
            car.get('소유자변경횟수')
        ]]
        
        # 2. 학습 시 사용한 다항 특성 변환기 적용
        list_poly = self.best_poly.transform(list_data)
        
        # 3. 학습 시 사용한 스케일러 적용 (있을 때만)
        if self.best_scaler is not None:
            list_poly = self.best_scaler.transform(list_poly)
            
        # 4. 최적 모델로 예측 수행
        result = self.best_model.predict(list_poly)
        
        # 5. 결과 반환 (만 단위 반올림)
        return {"예측매매가격": round(float(result[0]), 2)}

# 서비스 객체 생성 (컨트롤러에서 사용함)
car_service = CarService()
