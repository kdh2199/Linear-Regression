import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import platform

#운영체제에 맞는 글씨 깨짐 방지
def set_font():
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        return 'AppleGothic'
    elif platform.system() == "Windows":
        return 'Malgun Gothic'

#데이터셋 불러오기
df = pd.read_csv('student_20161231.csv', encoding='CP949')
#print(df)
#print(df.info())
#print(df.describe())

#데이터 추출
df1 = df[df.columns[2:4]]

#넘파이 배열에 데이터 저장
x = np.array(df[df.columns[3]])
y = np.array(df[df.columns[2]])

#학습용 데이터셋과 테스트용 데이터셋 구분
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scikit-learn 학습을 위해 행렬로 변환
X_train_skl = X_train.reshape(-1,1)
X_test_skl = X_test.reshape(-1,1)
y_train_skl = y_train.reshape(-1,1)
y_test_skl = y_test.reshape(-1,1)

#matplot 한글 꺠짐
rc('font', family=set_font())
plt.rcParams['axes.unicode_minus'] = False

#그래프로 출력하여 학습용 데이터 확인
plt.figure(figsize=(10, 5))

#자체 제작 알고리즘 학습용 데이터
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue')
plt.title('학습용 데이터 - 자체 제작 알고리즘')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()

#scikit-learn 알고리즘 학습용 데이터
plt.subplot(1, 2, 2)
plt.scatter(X_train_skl, y_train_skl, color='green')
plt.title('학습용 데이터 - scikit-learn 알고리즘')
plt.xlabel(df.columns[3], loc='right')
plt.grid()
plt.show()

#자체 제작 알고리즘 학습

#최초 직선 초기값 설정
x_mean = np.mean(X_train)
y_mean = np.mean(y_train)
length = len(X_train)
first_W = y_mean / x_mean
first_b = 1

#최초 직선 MSE 계산
total = 0
for i in range(length):
    total += ((X_train[i] * first_W + first_b) - y_train[i]) ** 2
MSE = total / length
#print("현재 기울기 : {0}, 현재 절편 : {1}".format(first_W, first_b))
#print("MSE = {0}".format(MSE))

#반복 시작
W = first_W
b = first_b
training = 0.0004
count = 0
while True:
    last_MSE = MSE
    total_w = 0
    total_b = 0
    SSE = 0

    #손실함수 계산
    for j in range(length):
        SE = (X_train[j] * W + b) - y_train[j]
        SSE += SE ** 2
        total_w += X_train[j] * SE
        total_b += SE
    
    #학습률을 사용하여 가중치와 편향 업데이트
    W -= training * (total_w * 2 / length)
    b -= training * (total_b * 2 / length)
    MSE = SSE / length
    
    #오차의 변화율
    delta = last_MSE - MSE
    
    #500000번 반복하거나 오차율의 변화가 일정 수치 밑으로 줄어들면 중지
    if count == 500000 or (abs(delta) < 0.0000000000001 and count > 1):
        #print("기울기 : {0} / 절편 : {1} / MSE : {2} / 차이 : {3:.15f}".format(W, b, MSE, delta))
        #print(count)
        break
    count += 1

#최종 MSE는 학습 데이터가 아닌 테스트 데이터로 계산
total = 0
for i in range(len(X_test)):
    total += ((W * X_test[i] + b) - y_test[i]) ** 2
MSE = total / len(X_test)

print("자체 제작 | 기울기 : {0} / 절편 : {1} / MSE : {2}".format(W, b, MSE))
#print("차이 : {0:.15f} / 반복 횟수 : {1}".format(delta, count))

#scikit-learn 알고리즘 학습

#선형회귀 모델 설정
model = LinearRegression()

#학습용 데이터셋으로 선형회귀 모델 학습
model.fit(X_train_skl, y_train_skl)

#테스트용 데이터셋으로 예측
y_pred = model.predict(X_test_skl)

#기울기, 절편, MSE 출력
mse = mean_squared_error(y_test_skl, y_pred)
slope = model.coef_[0][0]
intercept = model.intercept_[0]

print("scikit-learn | 기울기 : {0} / 절편 : {1} / MSE : {2}".format(slope, intercept, mse))

#두 모델 예측 결과 그래프로 출력
plt.figure(figsize=(10, 5))

#자체 제작 알고리즘 학습용 데이터
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, W * X_test + b, color='red')
plt.title('모델 학습 결과 - 자체 제작 알고리즘')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()

#scikit-learn 알고리즘 학습용 데이터
plt.subplot(1, 2, 2)
plt.scatter(X_test_skl, y_test_skl, color='green')
plt.plot(X_test_skl, y_pred, color='red')
plt.title('모델 학습 결과 - scikit-learn 알고리즘')
plt.xlabel(df.columns[3], loc='right')
plt.grid()
plt.show()