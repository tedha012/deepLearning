import numpy as np
import matplotlib.pyplot as plt

# 텐서플로의 케라스 API에서 필요한 함수
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 공부한 시간(x)와 점수(y)를 넘파이 배열로 선언
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

model = Sequential()

# 출력 값, 입력 변수, 분석 방법에 맞게 모델 설정
model.add(Dense(1, input_dim=1, activation="linear"))

# 오차 수정을 위해 경사 하강법(sgd)
# 오차의 정도를 판단하기 위해 평균 오차(mse)
model.compile(optimizer="sgd", loss="mse")

# 오차를 최소화하는 과정 2000번 반복
model.fit(x, y, epochs=2000)

plt.scatter(x, y)
print(model.predict(x))
plt.plot(x, model.predict(x), "r")
plt.show()
