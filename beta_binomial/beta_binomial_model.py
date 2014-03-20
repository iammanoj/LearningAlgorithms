import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import beta
import math
import scipy.stats as st

class BetaBinomial:
    """Beta Binomial Model.
    Use : 

    """    
    def __init__(self):
        self.a = None
        self.b = None
        
    def EmpiricalBayesEstimate(self,PriorData):
        
        '''Compute alpha and beta parameters using Method of Moments.

        Parameters
        ----------
            PriorData : 2D numpy array
                       Col 1 : Total N
                       Col 2 : Count of 1 
                       Count of 0 = Col_1 - Col_2
                       
        Returns
        -------
            alpha : float
            beta  : float
        
        '''
        if PriorData.shape[1] != 2:
            raise Exception("PriorData does not have 2 columns!")
        if sum(PriorData[:,0] < PriorData[:,1]) > 0:
            raise Exception("PriorData has first column has smaller number which is impossible!")
            
        Xi = np.zeros(PriorData.shape[0], dtype=float32)
        Xi[:] = PriorData[:,1]*1.0/PriorData[:,0]
        n = size(Xi)*1.0
        mu = sum(Xi)*1.0/n  # 1st Moment
        var = sum(np.square(Xi))*1.0/n   # 2nd Moment
        
        self.a = ( (1- mu)/var - 1/mu)* mu*mu
        self.b = self.a*(1/mu - 1)
        
        return(self.a,self.b)

    
    def PosteriorPredictive(self, newData, includeCI=False,confidence=0.95 , PriorData = None):
         
        '''Compute alpha and beta parameters using Method of Moments.

        Parameters
        ----------
            newData : 2D numpy array
                       Col 1 : Total N
                       Col 2 : Count of 1 
                       Count of 0 = Col_1 - Col_2
            PriorData : 2D numpy array, similar to newData

         Returns
        -------
             If includeCI=False, 1D numpy array with estimated theta (posterior)
             If includeCI=True,  2D numpy array with estimated theta (posterior) and Confidence Interval
        '''
        PriorData=None
        if PriorData is None and (self.a is None or self.a is None):
            raise Exception("Model parameters are not estimated!")
            
        if PriorData is not None:
            self.EmpiricalBayesEstimate(PriorData)
        
        # check if newData has right shape
        if newData.shape[1] != 2:
            raise Exception("newData does not have 2 columns!")
        if sum(newData[:,0] < newData[:,1]) > 0:
            raise Exception("First column has smaller number which is impossible!")
            
        if includeCI == True:
            out = np.zeros((newData.shape[0],2), dtype=float32)
            out[:,0] = (newData[:,1] + self.a) / ( newData[:,0]  + self.a + self.b)   
            z = st.norm.ppf(confidence)

            out[:,1] = z*np.sqrt(out[:,0]*(1-out[:,0])/newData[:,0])
            
        else:    
            out = np.zeros(newData.shape[0], dtype=float32)
            out[:] = ( newData[:,1] + self.a) / ( newData[:,0]  + self.a + self.b)     
        return out   


def main():
    
    # Test Prior Data    
    Prior =  np.zeros((10000,2))
    Prior[:,0] = np.random.randint(30, size=10000)
    Prior[:,1] = Prior[:,0] - np.random.randint(30, size=10000)
    Prior[np.where(Prior[:,1] < 0),1] = 0
    Prior[np.where(Prior[:,0] < 1),0] = 1
    # Test Data
    Data =  np.zeros((1000,2))
    Data[:,0] = np.random.randint(30, size=1000)
    Data[:,1] = Data[:,0] - np.random.randint(30, size=1000)
    Data[np.where(Data[:,1] < 0),1] = 0
    Data[np.where(Data[:,0] < 1),0] = 1

    # Train the model    
    BModel = BetaBinomial()
    BModel.EmpiricalBayesEstimate(Prior)
    print BModel.a, BModel.b
    
    # Get Prediction
    PredictiveEstimate = BModel.PosteriorPredictive(Data,includeCI=True )
    
if __name__ == "__main__":
    main()

