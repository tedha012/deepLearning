from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# iris 데이터를 불러옵니다.
df = pd.read_csv("./data/iris3.csv")

# 속성을 X, 클래스를 y로 저장
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# 원-핫 인코딩 처리를 합니다.
y = pd.get_dummies(y)

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=4, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

# 모델 컴파일
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# 모델 실행
history = model.fit(X, y, epochs=50, batch_size=5)
