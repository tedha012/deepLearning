import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


df = pd.read_csv("./data/pima-indians-diabetes3.csv")
x = df.iloc[:, 0:8]  # 세부 정보를 x로 지정합니다.
y = df.iloc[:, 8]  # 당뇨병 여부를 y로 지정합니다.

# 모델 구성
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu", name="Dense_1"))
# Input layer의 노드는 8개(파라미터 개수와 동일)
# Input layer 다음 첫번째 Hidden layer의 노드 : 12
# 활성화 함수(Input -> hidden layer1) : Rectified Linear Unit
# 해당 Dense의 이름을 Dense_1 으로 명명
model.add(Dense(8, activation="relu", name="Dense_2"))
# 두번째 Hiddenlayer의 노드 갯수 : 8
# 활성화 함수(hidden layer1 -> hidden layer2) : Rectified Linear Unit
# 해당 Dense의 이름을 Dense_2 으로 명명
model.add(Dense(1, activation="sigmoid", name="Dense_3"))
# 출력층(Output layer)의 노드 갯수 : 1
# 활성화 함수(hidden layer2 -> Output layer2) : sigmoid
# 해당 Dense의 이름을 Dense_3 으로 명명
model.summary()

# 모델 컴파일
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# 이진분류(당뇨냐 아니냐)문제이기에 로지스틱회귀에 사용된 바이너리 크로스엔트로피를 손실함수로 지정
# 옵티마이저(경사하강법을 적용하는 방식) : adam(교수님께서 가장 많이 쓴다고 말씀해주심)
# metrics(모델 수행의 결과를 나타내는 척도) : accuracy( 정확도)
# 모델 실행
history = model.fit(x, y, epochs=1000, batch_size=5)
