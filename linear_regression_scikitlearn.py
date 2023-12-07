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
x = np.array(df[df.columns[3]]).reshape(-1,1)
y = np.array(df[df.columns[2]]).reshape(-1,1)

#matplot 한글 꺠짐
rc('font', family=set_font())
plt.rcParams['axes.unicode_minus'] = False

#그래프 출력
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='green')
plt.title('학생들의 키와 몸무게')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()
plt.show()

#학습용 데이터셋과 테스트용 데이터셋 구분
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#선형회귀 모델 설정
model = LinearRegression()

#학습용 데이터셋으로 선형회귀 모델 학습
model.fit(X_train, y_train)

#테스트용 데이터셋으로 예측
y_pred = model.predict(X_test)

#기울기, 절편, MSE 출력
mse = mean_squared_error(y_test, y_pred)
slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(f'Mean Squared Error: {mse}')
print(f'기울기(계수): {slope}')
print(f'절편: {intercept}')

#그래프 출력
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red')
plt.title('학생들의 키와 몸무게')
plt.xlabel(df.columns[3], loc='right')
plt.ylabel(df.columns[2], rotation=0, loc='top')
plt.grid()
plt.show()