import scipy
import numpy as np
import pandas as pd
from pylab import *
from time import time


class  svm:
    def __init__(self):
        self.w = None
        self.b = None
        self.cost = None

    def cost_function(self,X,y,w,b):
        dist = 1 - y*(X.dot(w) + b)
        dist[dist <= 0] = 0
        L = sum(dist)
        return 0.5*w.dot(w) + self.cost*L
    
    def gradient_w(self,X,y,w,b):        
        if y*(X.dot(w) + b ) >= 1 :
            L = 0
        else:
            L = -y*X
        return w + self.cost *L
        
    def gradient_b(self,X,y,w,b):
        if y*(X.dot(w) + b ) >= 1 :
            return 0
        else:
            return -self.cost*y

    def train_sgd(self,X,y,ETA = 0.0001,epsilon = 0.001,cost = 100,trace=False):
        
        self.cost = cost        
        b = 0
        w = np.zeros(X.shape[1])
        
        # Start SGD
        n,d = X.shape
        ix = range(n)
        np.random.shuffle(ix)
        
        index = 0        
        i = ix[index]
        k = 0 
        delta_cost_old = 0.0
        delta_cost_new = epsilon + 1
        ret_k_cost = [[],[]]    
        t0 = time()
        
        while delta_cost_new > epsilon:
            
            w_old = w.copy()
            cost_old = self.cost_function(X,y,w_old,b)
            delta_cost_old = delta_cost_new
                 
            w = w_old - ETA*self.gradient_w(X[i,:],y[i],w_old,b)
            b = b - ETA*self.gradient_b(X[i,:],y[i],w_old,b)

            cost_new = self.cost_function(X,y,w,b)            
            cost_pct = abs(cost_old - cost_new)*100.0/cost_old
                        
            delta_cost_new = 0.5*delta_cost_old +0.5*cost_pct
       
            if trace == True:
                print k,delta_cost_old,delta_cost_new , cost_new
            index = (index+1) % n 
            i = ix[index]
            k = k+1
            ret_k_cost[0].append(k)
            ret_k_cost[1].append(cost_new)
        self.w = w
        self.b = b
        t1 = time()
        #print  'Time taken = %f sec' %(t1-t0)

        return ret_k_cost

    def train_GD(self,X,y,ETA = 0.0000003, epsilon = 0.25, cost = 100, trace=False ):
        
        # Initialize
        self.cost = cost     
        b = 0
        w = np.zeros(X.shape[1])
        
        # Start GD
        n,d = X.shape
        cost_pct = epsilon + 1
        k = 0
        ret_k_cost = [[],[]]   
        t0 = time()
        while cost_pct > epsilon :
            
            w_old = w.copy()
            cost_old = self.cost_function(X,y,w_old,b)            
            gr_w = np.zeros(X.shape[1])
            gr_b  =  0

            for i in range(n):
                gr_w = gr_w + self.gradient_w(X[i,:],y[i],w_old,b)
                gr_b = gr_b + self.gradient_b(X[i,:],y[i],w_old,b)
            
            w = w_old - ETA*gr_w
            b = b - ETA*gr_b
            cost_new = self.cost_function(X,y,w,b)       
            cost_pct = abs(cost_old - cost_new)*100.0/cost_old
            k = k+1
            if trace == True:
                print k,cost_old,cost_new , cost_pct
            ret_k_cost[0].append(k)
            ret_k_cost[1].append(cost_new)

        self.w = w
        self.b = b
        t1 = time()
        print  'Time taken = %f sec' %(t1-t0)
        return ret_k_cost
        
    def train_batch_GD(self,X,y,ETA = 0.00001, epsilon = 0.01, 
                       cost = 100,batch_size = 20,trace=False):
        # Initialize
        self.cost = cost     
        n,d = X.shape
        b = 0
        w = np.zeros(X.shape[1])
        l = 0
        k = 0
        ix = range(n)
        np.random.shuffle(ix)
        delta_cost_old = 0.0
        delta_cost_new = epsilon + 1
        ret_k_cost = [[],[]] 
        t0 = time()

        while delta_cost_new > epsilon :
            
            i = l*batch_size
            j = min(n-1,(l+1)*batch_size-1)
            gr_w = np.zeros(X.shape[1])
            gr_b  =  0
            w_old = w.copy()
            cost_old = self.cost_function(X,y,w_old,b)  
            delta_cost_old = delta_cost_new
            
            for index in range(i,j+1):
                m = ix[index]
                gr_w = gr_w + self.gradient_w(X[m,:],y[m],w_old,b)
                gr_b = gr_b + self.gradient_b(X[m,:],y[m],w_old,b)
            
            w = w_old - ETA*gr_w
            b = b - ETA*gr_b
            cost_new = self.cost_function(X,y,w,b)   
            cost_pct = abs(cost_old - cost_new)*100.0/cost_old
                           
            delta_cost_new = 0.5*delta_cost_old +0.5*cost_pct
            
            l = (l + 1) % (int(n + batch_size - 1)/batch_size)
            k = k + 1
            #print k,i, j, l
            if trace == True:
                print k,delta_cost_old,delta_cost_new , cost_new
            ret_k_cost[0].append(k)
            ret_k_cost[1].append(cost_new)
            
        self.w = w
        self.b = b
        t1 = time()
        print  'Time taken = %f sec' %(t1-t0)
        return ret_k_cost
        
    def predict(self,X):
         out = sum(self.w*X,axis=1) + self.b
         out[out >= 0] = 1
         out[out < 0] = -1
         return out

 
        
    
'''
X_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/features.txt'
y_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/target.txt'

XX = np.array(pd.read_csv(X_file,delimiter =',', header=None))
yy = np.array(pd.read_csv(y_file,delimiter =',', header=None)).flatten()

sgd_svm = svm()
out_sgd = sgd_svm.train_sgd(XX,yy)
plot(out_sgd[0],out_sgd[1],color='b',label='sgd')

bgd_svm = svm()
out_bgd = bgd_svm.train_batch_GD(XX,yy)
plot(out_bgd[0],out_bgd[1],color='r',label='bgd')

gd_svm = svm()
out_gd = gd_svm.train_GD(XX,yy)
plot(out_gd[0],out_gd[1],color='g',label='gd')

show()
'''
#########

X_train_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/features.train.txt'
y_train_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/target.train.txt'

X_test_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/features.test.txt'
y_test_file = '/Users/sdey/Documents/cs246/home-work/hw04/HW4-q1/target.test.txt'


X_train = np.array(pd.read_csv(X_train_file,delimiter =',', header=None))
y_train = np.array(pd.read_csv(y_train_file,delimiter =',', header=None)).flatten()

X_test = np.array(pd.read_csv(X_test_file,delimiter =',', header=None))
y_test = np.array(pd.read_csv(y_test_file,delimiter =',', header=None)).flatten()


cost_list = [ 1,10, 50, 100, 200, 300, 400, 500]

sgd_svm = svm()
cost_error =  [[],[]] 
for c in cost_list:
    out_sgd = sgd_svm.train_sgd(X_train,y_train, ETA = 0.0001,epsilon = 0.001, cost=c)
    y_test_hat = sgd_svm.predict(X_test)
    err = sum(y_test_hat <> y_test)*1.0/size(y_test)
    print 'Cost = ',c, ' error =',err
    cost_error[0].append(c)
    cost_error[1].append(err)

plot(cost_error[0],cost_error[1],color='g')
show()



