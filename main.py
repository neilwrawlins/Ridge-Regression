from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

#make_regression generates a random regression problem
X, y, coefficients = make_regression(
    n_samples=50, #number of samples (default = 100)
    n_features=1, #number of features (default = 100)
    n_informative=1, #The number of informative features, i.e., the number of features used to
                     #build the linear model used to generate the output. (default = 10)
    n_targets=1, #The number of regression targets, i.e., the dimension of the y output vector
                 #associated with a sample. By default, the output is a scalar. (default = 1)
    noise=5, #The standard deviation of the gaussian noise applied to the output. (default = 0.0)
    coef=True, #If True, the coefficients of the underlying linear model are returned. (default = False)
    random_state=1 #Determines random number generation for dataset creation.
)


#This is identitical to linear regression. A high alpha makes bias high.
rr = Ridge(alpha = 1)
rr.fit(X,y)
w = rr.coef_[0]
plt.scatter(X, y)
#regression line, will be same as linear regression since alpha is 1
plt.plot(X, X*w, c='red')
plt.show()

#Increasing alpha gives us a less steep slope which increases bias a little to hopefully
#decrease variance by a lot
rr = Ridge(alpha=10)
rr.fit(X, y)
w = rr.coef_[0]
plt.scatter(X, y)
plt.plot(X, X*w, c='red')
plt.show()

rr = Ridge(alpha=100)
rr.fit(X, y)
w = rr.coef_[0]
plt.scatter(X, y)
plt.plot(X, X*w, c='red')
plt.show()