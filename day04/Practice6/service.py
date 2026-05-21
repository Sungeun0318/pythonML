# PythonML Practice 6: 중고차량 매매가 예측 시스템 구축
# [상세 요구사항]
# 1. 중고차량의 사양 및 거래 가격 정보를 포함한 샘플 데이터를 데이터베이스(DB)에 적재 및 관리한다.
# 2. 관리자가 Spring Boot REST API를 호출하여, 데이터베이스에 저장된 전체 데이터를 기반으로 예측 모델을 (재)학습하고 최신화할 수 있어야 한다.
# 3. 일반 사용자가 REST API를 통해 5개 핵심 변수(평균 연비, 누적 주행거리(km), 출고 후 경과 월수, 사고 감가 건수, 소유자 변경 횟수*)를 입력하여 요청하면, 모델이 예측한 매매 가격(단위: 만 원)을 실시간으로 반환한다.
# 4. 학습된 예측 모델의 예측 정밀도를 보장하기 위해 평가지표인 결정계수는 최소 90% 이상(0.90 이상)을 달성해야 한다.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class CarService:
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaler = None

    def train(self, carList):
        df = pd.DataFrame(carList)
        x = df[['평균연비', '누적주행거리키로', '출고후경과월수', '사고감가건수', '소유자변경횟수']]
        y = df['매매가격만원'].values
       
        train_input, test_input, train_target, test_target = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        

        optimization = []
        
        for degree in [1, 2, 3]:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            train_poly = poly.fit_transform(train_input)
            test_poly = poly.transform(test_input)
            
            lr = LinearRegression()
            lr.fit(train_poly, train_target)
            r2 = lr.score(test_poly, test_target)
            optimization.append({'r2': r2, 'model': lr, 'poly': poly, 'scaler': None})
            ss = StandardScaler()
            train_scaled = ss.fit_transform(train_poly)
            test_scaled = ss.transform(test_poly)
            
            for alpha in [0.01, 0.1, 1, 10, 100]:
                ridge = Ridge(alpha=alpha)
                ridge.fit(train_scaled, train_target)
                r2 = ridge.score(test_scaled, test_target)
                optimization.append({'r2': r2, 'model': ridge, 'poly': poly, 'scaler': ss})
                
                lasso = Lasso(alpha=alpha)
                lasso.fit(train_scaled, train_target)
                r2 = lasso.score(test_scaled, test_target)
                optimization.append({'r2': r2, 'model': lasso, 'poly': poly, 'scaler': ss})
        
        best_optimization = max(optimization, key=lambda x: x['r2'])
        
        self.best_model = best_optimization['model']
        self.best_poly = best_optimization['poly']
        self.best_scaler = best_optimization['scaler']
        
        print(f"학습 완: {best_optimization['r2']}")

    def predict(self, car):
        if self.best_model is None:
            return {"학습 안됨"}
        
        list_data = [[
            car.get('평균연비'), 
            car.get('누적주행거리키로'), 
            car.get('출고후경과월수'), 
            car.get('사고감가건수'), 
            car.get('소유자변경횟수')
        ]]
        
        list_poly = self.best_poly.transform(list_data)
        
        if self.best_scaler is not None:
            list_poly = self.best_scaler.transform(list_poly)
            
        result = self.best_model.predict(list_poly)
        return int(result[0])


car_service = CarService()