import numpy as np
import matplotlib.pyplot as plt

# 텐서플로의 케라스 API에서 필요한 함수
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 공부한 시간(x)와 점수(y)를 넘파이 배열로 선언
x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1])

model = Sequential()

# 출력 값, 입력 변수, 분석 방법에 맞게 모델 설정
model.add(Dense(1, input_dim=1, activation="sigmoid"))

# 교차 엔트로피 오차 함수를 이용하기 위해 'binary_crossentropy'로 설정
model.compile(optimizer="sgd", loss="binary_crossentropy")

# 오차를 최소화하는 과정 5000번 반복
model.fit(x, y, epochs=5000)

plt.scatter(x, y)
print(model.predict(x))
plt.plot(x, model.predict(x), "r")
plt.show()

# 임의의 학습 시간을 집어넣어 합격 예상 확률을 예측해 보겠습니다.
hour = 7
prediction = model.predict([hour])
print(f"{hour} 시간을 공부할 경우, 합격 예상 확률은 {prediction*100}% 입니다.")
