"""
Problem Set # 3 K-means
@author: sdey
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def distance(x1,x2):
    return sqrt(sum(np.square(x1-x2)))
    
def Kmeans(X,k,iter):
    m =  X.shape[0]
    
    # Create temporayli array to keep the mapping between points and clusters
    cluster_assignment = np.zeros((m),dtype=int)  
        
    # Randomly initialize the cluster centroids                  
    cluster_ix = np.random.randint(1, high=m, size=k)
    clusters = np.zeros((k,X.shape[1]),dtype=double)
    clusters[:,:] = X[cluster_ix ,:]
    
    for i in range(iter):
        # Assign the clusters to the X points
        for j in range(m):
            dist = sys.maxint
            cluster_assignment[j] = 0
            for l in range(k):
                if dist >  distance(X[j,:],clusters[l,:]):
                    cluster_assignment[j]  = l
                    dist = distance(X[j,:],clusters[l,:])
        # Now Calulate the centroids    
        for n in range(k): 
            ix = np.asarray(np.where(cluster_assignment == n)).flatten()
            clusters[n,:] = np.mean(X[ix,:],axis=0)
    
    return [cluster_assignment,clusters]
    
   
   # Main Block        

if __name__ == "__main__":
    # Define Image file
  #image_file='/Users/sdey/Documents/cs229/Assignments/ps3/mandrill-large.tiff'
    image_file='/Users/sdey/Documents/cs229/Assignments/ps3/mandrill-small.tiff'
    
    # Read Image Files    
    img = misc.imread(image_file )   
    
    
    # Display the image
    print 'Original Image'    
    imgplot_o = plt.imshow(img)
    
    # Transform into 2D Matrix    
    X = np.ravel(img[:,:,:]).reshape(img.shape[0]*img.shape[1],img.shape[2])
       
    # Set parameters  
    k = 16
    iter = 30
    
    # Call K-Means now 
    [x_cluster_map,clusters_centroids]= Kmeans(X,k,iter)
    
    #Assign the the values to cluster centroids 
    X_new = np.zeros(X.shape,dtype=uint8)
    for j in range(X.shape[0]):
        X_new[j,:] = clusters_centroids[x_cluster_map[j],:] 
        
    #Tranform into 3D Array to rebuild the image
    new_img = np.ravel(X_new[:,:]).reshape(img.shape[0], img.shape[1], 
                                            img.shape[2])
    
    #Displya the new image
    print 'Compressed Image'
    imgplot_n = plt.imshow(new_img)
