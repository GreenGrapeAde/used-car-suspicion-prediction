---

---

[다른 파일에서 사용할 때]

from inference import SuspectedCarPredictor
import pandas as pd

df = pd.read_csv("used_cars_data.csv").tail(3)

predictor = SuspectedCarPredictor(
    model_path="./weights_005_0.38692.pth",
    scaler_path="./scaler_juyeong.pkl",
    cat_mapping_path="./cat_mapping.pkl",
    config_path="./model_config.pkl"
)

result = predictor.predict(df, threshold=0.6)
print(result[["suspected_prob", "suspected_pred"]].head())

---

데이터 출처: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset

---

## 📊 최종 인사이트

- 모델의 파라미터 조정도 중요하지만, 근본적으로는 데이터의 특성과 구조를 이해하는 과정이 더욱 중요하다는 점을 확인하였습니다.
- 또한 데이터의 특성에 따라 적절한 모델과 전처리 방식을 선택하는 것이 성능에 큰 영향을 미친다는 점을 학습하였습니다.
- 본 프로젝트는 하나의 정답을 도출하는 것보다, 데이터 전처리·Feature 선택·모델 설계 과정에서 무엇을 고려해야 하는지를 경험하는 데 의미가 있었습니다.
- 향후에는 데이터 정제 및 모델 고도화를 통해 국내 중고차 시장 환경에 맞는 서비스로 확장 가능성을 검토할 수 있습니다.