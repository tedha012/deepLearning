from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("./data/sonar3.csv", header=None)

# 음파 관련 속성을 X, 광물의 종류를 y
X = df.iloc[:, 0:60]
y = df.iloc[:, 60]

# 학습셋과 테스트셋을 구분
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 실행
history = model.fit(X, y, epochs=300, batch_size=50)

# 모델을 테스트셋에 적용해 정확도를 구함
score = model.evaluate(X_test, y_test)
print("Test accuracy : ", score[1])

# 모델 저장
model.save("./data/model/my_model.hdf5")
