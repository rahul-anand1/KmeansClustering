# Importing required libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Kmean:
    def __init__(self):
        self.plot = True  # True if you want to plot final result else False
        self.data = None  # If entire list is to be used update the list here
        if self.data == None or len(self.data) == 0:
            self.data = np.random.rand(100,2)

    def operation(self):
        '''
        Main operation of K means cluster.
        '''
        k = 3 # the number of clusters to be made
        iterations = 10 # the number of iterations updating centroids
        self.data = pd.DataFrame(self.data)
        X = np.array(self.data)
        # Initialization of random centroids
        init_c = random.sample(range(0, len(self.data)), k)
        # Creating the list of centroids
        centroids = []
        for i in init_c:
            centroids.append(self.data.loc[i])
        centroids = np.array(centroids)

        for i in range(iterations):
            get_centroids = self.__findClosestCentroids__(centroids, X)
            centroids = self.__calc_centroids__(get_centroids, X)

        final_df = pd.concat([pd.DataFrame(X), pd.DataFrame(get_centroids, columns=['Cluster'])],
                          axis=1)
        final_df.to_csv('Result.csv', index = False)
        if self.plot == True:
            self.__plot__(centroids, X)
        return True

    def __calculate_distance__(self, x, y):
        '''
        Calculating the distance between two data points.
        '''
        return(sum((x - y)**2))**0.5

    def __findClosestCentroids__(self, c, X):
        '''
        Finding the closest centroid based on the distance between
        centroid and the data point.
        '''
        assigned_centroid = []
        for i in X:
            distance=[]
            for j in c:
                distance.append(self.__calculate_distance__(i, j))
            assigned_centroid.append(np.argmin(distance))
        return assigned_centroid

    def __calc_centroids__(self, clusters, X):
        '''
        Updating the centroids based on the mean distance of the data
        points belonging to a cluster.
        '''
        new_centroids = []
        new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],
                          axis=1)
        for c in set(new_df['cluster']):
            current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
            cluster_mean = current_cluster.mean(axis=0)
            new_centroids.append(cluster_mean)
        return new_centroids

    def __plot__(self, centroids, X):
        '''
        Plotting the final cluster using matplotlib
        '''
        fig = plt.figure()
        plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
        plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
        plt.show()
        fig.savefig('cluster.jpeg')


if __name__ == "__main__":
    kobj = Kmean()
    result = kobj.operation()
