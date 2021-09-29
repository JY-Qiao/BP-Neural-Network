import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from multilayer_perceptron import MultilayerPerceptron


data = pd.read_csv('../data/wheel_speed.csv')

train_data = data.sample(frac = 0.9)
test_data = data.drop(train_data.index)

train_data = train_data.values
test_data = test_data.values


x_train = train_data[:,:2]
y_train = train_data[:,[2]]

x_test = test_data[:,:2]
y_test = test_data[:,[2]]

# layers = [2,1]
layers=[2,15,1]

normalize_data = True
max_iterations = 500
alpha = 0.01


multilayer_perceptron = MultilayerPerceptron(x_train,y_train,layers,normalize_data)
(thetas,costs) = multilayer_perceptron.train(max_iterations,alpha)

plt.plot(range(len(costs)),costs)
plt.xlabel('Grident steps')
plt.ylabel('costs')
plt.show()


y_train_predictions = multilayer_perceptron.predict(x_train)
y_test_predictions = multilayer_perceptron.predict(x_test)

print('Test set prediction results：',y_test_predictions)
print('Test set precision results：',y_test)
print('Train set prediction results：',y_train_predictions)
print('Train set precision results：',y_train)

train_p = (np.sum((y_train_predictions - y_train) ** 2)/y_train.shape[0]) ** 0.5
test_p = (np.sum((y_test_predictions - y_test) ** 2)/y_test.shape[0]) ** 0.5
print ('Train set standard deviation：',train_p)
print ('Test set standard deviation：',test_p)