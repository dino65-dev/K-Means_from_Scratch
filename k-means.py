import random
import numpy as np


class K_means:
    """
      ** I've given you the examples comment based on 2D data **

    args: 
    n_clusters = How much clusters needed to make for the data , By default it is 2.
    
    max_iter = The number of interations the clustering needs for properly cluster the data points, By default it is 100.
    
    centroids = It will automatically create centroids for the data given


     """
    
    def __init__(self,n_clusters = 2, max_iter = 120) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    
    def fit_predict(self,X):
         random_index = random.sample(range(0,X.shape[0]),self.n_clusters) # creating random coordinates for to place the centroids
         self.centroids = X[random_index] # placing the centroids at the random_index or coordinates
         
         for i in range(self.max_iter): # running the loop for given max_iter value to move the centroids and the data points
             
             #assign clusters
             cluster_group = self.assign_clusters(X)
             old_centroids = self.centroids # this represents the old cluster
             
             #move centroids
             self.centroids = self.move_centroids(X,cluster_group) # this represnts the new cluster that is assinged
             
             #check finish
             if (old_centroids == self.centroids).all():
                 break
             
         return cluster_group

    
    def assign_clusters(self,X):
          cluster_group = [] # it stores the coressponding cluster groups depending on the index_pos (eg. 1,0,0,1[if there is only two centroids]) like this.
          distances = [] # it stores the Ecluidean Distances btwn the data points so we can determine which is the closest centroid

          for row in X: 
              for centroid in self.centroids: # this will run for the centroid*num_of_colmn_in_a_row (means for each data points in a centroid eg. 2 centroids and 100 data points so it will run 200 times)
                  distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
              min_distance = min(distances) # the min distance of from each centroid (eg. if there is 2d array in X so lets take [2.433,46.5634] -
              # - so it will take the 2.433 that means the 1st centroid is the closest)
              index_pos = distances.index(min_distance) # this gives us the min_distances index which is the closest centroid
              cluster_group.append(index_pos)
              distances.clear()
          return np.array(cluster_group)

      
    def move_centroids(self,X,cluster_group): # it will group the centroids and calculate their means
       new_centroids = [] # it will store the mean value that is the new centroids

       unique_clusters = np.unique(cluster_group) # it will give the number of unique cluster (eg. 1,0,3 [so there is 3 unique clusters])

       for type in unique_clusters: # it will run the number of unique clusters
           new_centroids.append(X[cluster_group == type].mean(axis=0)) # it will calculate the mean value of the array according to centroid value that will be our new centroids 
        
       return np.array(new_centroids)
