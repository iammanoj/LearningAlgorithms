################################################################################

#    PROBLEM 2D

################################################################################


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg


class LinearRegression:
    """
    This is the class for linear regression
    """
    def __init__(self):
        self.theta = None
  
    def train(self, y,X):
        theta = np.zeros(X.ndim+1)
        # Add a new column with default value as 1 for the intercept
        if len(X.shape) == 1:
            X.shape = (X.shape[0],1)
            
        X_updated = np.append(np.ones([len(X),1]),X,1)

        # Normal equation         
        theta = linalg.inv(X_updated.transpose().dot(X_updated)).dot(X_updated.transpose().dot(y))
        self.theta = theta 
        
    def predict(self, X):
        
        if len(X.shape) == 1:
            X.shape = (X.shape[0],1)
        X_updated = np.append(np.ones([len(X),1]),X,1)
        return  X_updated.dot(self.theta)

 
class LocallyWeighterRegression:
    """
    This is the class for locally weighted linear regression
    """
    def __init__(self,X,y):
        
        # Add a new column with default value as 1 for the intercept
        if len(X.shape) == 1:
            X.shape = (X.shape[0],1)

        X_updated = np.append(np.ones([len(X),1]),X,1)
        
        # Store both X and y as part of the class. They will be used during prediction
        self.X = X_updated.copy()
        self.y = y
        self.m = X.shape[0]
        
    def getWeightsMatrix(self, X_new, tau):
        w = np.zeros([self.m,self.m])
        for j in range(self.m):
            w[j,j] = np.exp((-1)*((1/(2*np.power(tau,2)))*(X_new - self.X[j]).transpose().dot(X_new - self.X[j])))/2
        #print w        
        return w
        
    def predict(self, X_new,tau=2):
        if len(X_new.shape) == 1:
            X_new.shape = (X_new.shape[0],1)
            
        X_updated = np.append(np.ones([len(X_new),1]),X_new,1)
        
        out = np.array([])
              
        for i in range(len(X_new)):
            W = self.getWeightsMatrix(X_updated[i,:],tau)
            theta = linalg.inv(self.X.transpose().dot(W).dot(self.X)).dot(self.X.transpose().dot(W).dot(self.y))
            out = np.append(out,X_updated[i,:].dot(theta))

        return out

 
if __name__ == "__main__":
    
    X = np.loadtxt('/Users/sdey/Documents/cs229/Assignments/q2x.dat')
    y = np.loadtxt('/Users/sdey/Documents/cs229/Assignments/q2y.dat')
    
    lm = LinearRegression()
    lm.train(y,X)
    ybar = lm.predict(X)
    print lm.theta
    
    plt.plot(X, y,'bo')
    #plt.plot(X, ybar, color ='y')
    plt.title("Problem 2(d), i")
    #plt.show()
    #plt.plot(X, y,'bo')
    lwm = LocallyWeighterRegression(X,y)    
    #X_sorted = np.sort(X,axis=0)    
    X_sorted = np.arange(min(X),max(X),0.1)
    #tau_list = [(0.1,'red'), (0.3,'b'),(0.8,'g'),(2,'m'), (10,'k')]    
    tau_list = [0.1, 0.3, 0.8,2,10]    

    for t in tau_list:    
        Yhat = lwm.predict(X_sorted,tau=t)
        print 'Tau : %2.1f RMSE : %1.4f '% (t, np.power(np.sum(np.square(y-lwm.predict(X,tau=t)))/len(y),0.5))
        plt.plot(X_sorted, Yhat)
    
    plt.title("Problem 2(d), ii")
    plt.show()
    
    #[ 0.32767539  0.17531122]
    #Tau : 0.1 RMSE : 0.1044 
    #Tau : 0.3 RMSE : 0.1344 
    #Tau : 0.8 RMSE : 0.1541 
    #Tau : 2.0 RMSE : 0.8165 
    #Tau : 10.0 RMSE : 0.8165 
    
    
    
    
    