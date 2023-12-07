import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
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
#print(x)
#print(y)
#print(type(x))

#matplot 한글 꺠짐
rc('font', family=set_font())
plt.rcParams['axes.unicode_minus'] = False

#그래프 그려보기
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue')
plt.title('학생들의 키와 몸무게')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()
#plt.show()

#초기 가중치와 편향을 위해 x, y값의 평균 구하기
x_mean = np.mean(x)
y_mean = np.mean(y)
length = len(x)
#print(x_mean, y_mean)

#초기값 설정
first_W = y_mean / x_mean
first_b = 1

#그래프 그리기
plt.ylim(np.min(y)-10, np.max(y)+10)
plt.plot(x, x * first_W + first_b, color="red", linewidth=2)
plt.show()

#MSE 계산
total = 0
for i in range(length):
    total += ((x[i] * first_W + first_b) - y[i]) ** 2
MSE = total / length
print("현재 기울기 : {0}, 현재 절편 : {1}".format(first_W, first_b))
print("MSE = {0}".format(MSE))

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
        SE = (x[j] * W + b) - y[j]
        SSE += SE ** 2
        total_w += x[j] * SE
        total_b += SE
    
    #학습률을 사용하여 가중치와 편향 업데이트
    W -= training * (total_w * 2 / length)
    b -= training * (total_b * 2 / length)
    MSE = SSE / length
    
    #오차의 변화율
    delta = last_MSE - MSE
    
    #중간값 확인
    if count%10000 == 0:
        print("기울기 : {0} / 절편 : {1} / MSE : {2} / 차이 : {3:.15f}".format(W, b, MSE, delta))

    #500000번 반복하거나 오차율의 변화가 일정 수치 밑으로 줄어들면 중지
    if count == 500000 or (abs(delta) < 0.0000000000001 and count > 1):
        print("기울기 : {0} / 절편 : {1} / MSE : {2} / 차이 : {3:.15f}".format(W, b, MSE, delta))
        print(count)
        break
    count += 1
print("기울기 : {0} / 절편 : {1} / MSE : {2}".format(W, b, MSE))
print("차이 : {0:.15f} / 반복 횟수 : {1}".format(delta, count))

#그래프 그리기
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue')
plt.title('학생들의 키와 몸무게')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()

plt.ylim(np.min(y)-10, np.max(y)+10)
plt.plot(x, x * W + b, color="red", linewidth=2)
plt.show()