################################################################################

#    PROBLEM 1B

################################################################################

import numpy as np
from math import exp
from numpy import vectorize
import matplotlib.pyplot as plt
from numpy import linalg

def prob_threasholds(z):
    if z > 0.5 :
        return 1
    else:
        return 0

classify = vectorize(prob_threasholds)

def exponential(z):
    return (1/(1+exp((-1)*z)))

logit = vectorize(exponential)

def hessian(X,theta):
    A = np.zeros((len(theta),len(theta)),dtype= np.float64)
    for i in range(len(theta)):
        for j in range(len(theta)):
          #  print i,j
            sum = 0.0
            for k in range(X.shape[0]):
                sum = sum + (-1)*X[k,i]*X[k,j]*logit(theta.dot(X[k,:]))*(1-logit(theta.dot(X[k,:])))
        
            A[i,j] = sum
    return A


class LogisticRegression:
    """
    This is the main class
    """
    def __init__(self):
        self.num_iteration = None
        self.learning_rate = None
        self.theta = None

    def train(self, y,X,type='GRADIENT',learning_rate = 0.01,num_iteration = 1000):
        # Initialize theta with zeros
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        theta = np.zeros(X.shape[1]+1)
        # Add a new column with default value as 1 for the intercept
        if len(X.shape) == 1:
            X.shape = (X.shape[0],1)
        X_updated = np.append(np.ones([len(X),1]),X,1)

        # start gradient descent
        if type == 'GRADIENT':
            for i in range(self.num_iteration):
                yprime = y - logit(X_updated.dot(theta))
                theta = theta + self.learning_rate*yprime.dot(X_updated)

            # Store the final theta
            self.theta = theta

        # Newton Method
        if type == 'NEWTON':
            for i in range(self.num_iteration):
                yprime = y - logit(X_updated.dot(theta))
                #theta = theta + self.learning_rate*yprime.dot(X_updated)
                theta = theta -  linalg.inv(hessian(X_updated,theta)).dot(yprime.dot(X_updated))
                #print i,theta
            self.theta = theta
            

    def predict(self, X):
        X_updated = np.append(np.ones([len(X),1]),X,1)
        return  classify(logit(X_updated.dot(self.theta)))


if __name__ == "__main__":
    Xraw = np.loadtxt('/Users/sdey/Documents/cs229/Assignments/q1x.dat')
    y = np.loadtxt('/Users/sdey/Documents/cs229/Assignments/q1y.dat')

    model = LogisticRegression()
    model.train(y,Xraw,type='NEWTON',num_iteration = 10)
    print 'Parameters are : ',model.theta
    Yhat = model.predict(Xraw)
    print 'Misclassification  Rate: %1.2f' % (np.sum(abs(y-Yhat))/len(y))
    plt.plot(Xraw[np.where(y==1),0],Xraw[np.where(y==1),1], 'bo-')
    plt.plot(Xraw[np.where(y==0),0],Xraw[np.where(y==0),1], 'ro-')

    x_array = []
    y_array = []    
    #for i in range(10):
    for i in np.arange(min(Xraw[:,0]),max(Xraw[:,0]),0.1):
        x_array.append(i)
        y_array.append(((-1)*model.theta[0] + (-1)*model.theta[1]*i)/model.theta[2] )
    #print x_array,y_array    
    plt.plot(x_array,y_array)
    plt.title("Problem 1b")
    plt.show()
    
    
    
    
        
    
    







