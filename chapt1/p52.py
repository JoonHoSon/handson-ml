# coding: utf-8

__author__ = 'JoonHo Son'
__email__ = 'joonho.son@me.com'

# 핸즈온 머신러닝 1권 52페이지

import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression  # 선형 회귀
from sklearn.neighbors import KNeighborsRegressor

if len(sys.argv) != 2:
    print('usage : python p52.py [k | l]')
    exit(1)

model_type = sys.argv[1].lower()

if 'k' != model_type and 'l' != model_type:
    print('usage : python p52.py [k | l]')
    exit(1)

# 데이터를 다운로드하고 준비합니다.
data_root = 'https://github.com/ageron/data/raw/main/'
lifesat = pd.read_csv(data_root + 'lifesat/lifesat.csv')
x = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

# 데이터를 그래프로 나타냅니다.
lifesat.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction')
plt.axis([23_500, 62_500, 4, 0])
plt.show()

# 선형 모델을 선택합니다.
# 선형 회귀 모델일 경우
model = LinearRegression() if 'l' == model_type else KNeighborsRegressor(n_neighbors=3)

# k-최근접 이웃 회귀
# model = KNeighborsRegressor(n_neighbors=3)

# 모델을 훈련합니다.
model.fit(x, y)

# 키프로스에 대해 예측을 만듭니다.
x_new = [[37_655.2]]  # 2020년 키프로스 1인당 GDP

# KNeighborsRegressor : 6.33333333
# LinearRegression : 6.30165767

print(model.predict(x_new))
