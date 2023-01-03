import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as sk
df = pd.read_csv("insurance.csv")
df.drop(['sex', 'bmi', 'children', 'smoker', 'region'], axis=1, inplace=True)
x, y = df['age'], df['charges']
a = df.shape
print(a)
# Regression review
reg = sk.LinearRegression()
R = reg.fit(df[['charges']], df['age'])
reg_c = reg.coef_
reg_i = reg.intercept_

# df['age'].dtype, df['charges'].dtype #to check the datatype
df['age'].astype(float)
a = df.isnull().values.any()
print(a)

plt.scatter(df['charges'], df['age'], marker='+', color='blue')
plt.plot(df[['charges']], reg.predict(df[['charges']]),color="black",linewidth=3)
plt.xlabel('Charges')
plt.ylabel('Age')
plt.show()

