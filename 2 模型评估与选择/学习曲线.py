from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 10)


train_score = []
test_score = []


# 绘制学习曲线
for i in range(1,76):
    lin_reg = LinearRegression()
    lin_reg.fit(x_train[:,i],y_train[:,i])

    y_train_predict = lin_reg.predict(x_train[:,i])
    train_score.append(mean_squared_error(y_train[:,i],y_train_predict))

    y_test_predict = lin_reg.predict(x_test)
    test_score.append(mean_squared_error(x_test,y_test_predict))