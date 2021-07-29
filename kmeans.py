import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd

#create distance function
def eu_distance(A,B):
    
    #distance formula
    distance = math.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
    return distance

#create a function that reads csv files
def reader(x):
    
    #empty list
    data = {}
    #open in read mode
    with open(x, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter =",")
        #for each row
        for row in reader:
            #while ignoring the first row
            if row[0] != "Countries":
                #populate dictionary with keys, and tuple as values
                data[row[0]] = (float(row[1]),float(row[2]))
        return data


#create functions to calculate distance from points to centroid
def dcentroid(Dict,centroid_dict):
    
    #create empty dictionaries
    clust_dic = {}
    centroid_nearest_neighbors = {}
    dist_sum = 0
    #choose closest centroid at random
    
    #define the minimum distance to be the distance from a random country to the "closest" centroid
    for key in Dict:
        closest = random.choice(list(centroid_dict))
        min_distance = 999999999999999999
        for centroid in centroid_dict:
            #calculate distances to each centroid in the list of centroids
            dist = eu_distance(Dict[key],centroid_dict[centroid])
            #if there is a smaller distance than the minDistance
            if dist < min_distance:
                #replace with new closest
                closest = centroid
                #new mnimum distance = distance
                min_distance = dist

        #assign to dictionary
        if centroid_dict[closest] in centroid_nearest_neighbors:
            centroid_nearest_neighbors[centroid_dict[closest]].append(Dict[key])
        else:
            centroid_nearest_neighbors[centroid_dict[closest]] = [Dict[key]]

        #creates a dictionary for clusters to use to find which countries belong to which cluster as opposed to coordinates
        if centroid_dict[closest] in clust_dic:
            clust_dic[centroid_dict[closest]].append(key)
        else:
            clust_dic[centroid_dict[closest]] = [key]

    #return two objects           
    return centroid_nearest_neighbors, clust_dic
   

#create a two dimensional mean function
def twod_mean(Dict):
    
    #takes in a dictionary as the argument and splits the X and Y components
    sum_X = 0
    sum_Y = 0
    #initialize a counter
    counter = 0
    #create an empty array
    means = []
    
    for key in Dict:
        
        for i in Dict[key]:
            sum_X += i[0]
            sum_Y += i[1]
            counter +=1
            mean_X = sum_X/(counter)
            mean_Y = sum_Y/counter
        temp = (mean_X, mean_Y)
        means.append(temp)
        
    return(means)
      

#create a function that visualizes the clusters and assigns them a unique color
def visualize_cluster(clusters,Counter):
    #use a seed to make sure the colours remain the same throughout the iterations
    random.seed(Counter)
    X = []
    Y = []
    #create colors chosen at random, at max 3 because max 4 clusters.
    r = random.random()
    b = random.random()
    g = random.random()
    a = random.random()
    color = (r,g,b,a)
    #split into x and y values
    for i in clusters:
        X.append(i[0])
        Y.append(i[1])
    #plot on graph
    plt.scatter(X, Y, c = np.array([color]))
    
#define kmeans main algorithm    
def kmeans(filename):
    
    #initialize distSum
    dist_sum = 0
    d = []
    num_centroid = int(input("How many centroids to start? "))
    #read the data
    data = reader(filename)
    #initialize random centroids
    random.seed(0)
    centroid_country = (random.sample(list(data),num_centroid))
    centroid_dict = {}
    numClust = []
    
    for i in range(len(centroid_country)):
        centroid_dict[i] = data[centroid_country[i]]

    #loop the function according to user input
    num_iterations = int(input("how many iterations to start: "))
    
    for i in range(num_iterations):
        #call the function that calculates the distance of the closest centroid
        centroid_dic = dcentroid(data,centroid_dict)[0]
        for key in centroid_dic:
            for value in centroid_dic[key]:
                distance_squared = eu_distance(value,key)**2
                dist_sum += distance_squared
        print("\ncalculating the sum of the distances squared, checking for convergence: ", dist_sum)
        clust_dic = dcentroid(data,centroid_dict)[1]
        #calculate the new centroid
        new_centroid = twod_mean(centroid_dic)
        #clear the centroidDict
        centroid_dict.clear()
        #populate it with the new means as the centroids
        for i in range(len(centroid_country)):
            centroid_dict[i] = new_centroid[i]
        #initialize a Counter
        Counter =  0
        #plot the clusters for each key
        for key in centroid_dic:
            visualize_cluster(centroid_dic[key], Counter)
            Counter +=1
    plt.show()
    for key in centroid_dict:
        print("\nThe mean Birth rate for cluster", key ,"is: ", centroid_dict[key][0])
        print("The mean Life expectancy for cluster", key, "is: ", centroid_dict[key][1]) 
    print("\nThe number of countries belonging to each cluster are:\n")
    for key in clust_dic:
        print(len(clust_dic[key]))
    df = pd.DataFrame.from_dict(clust_dic, orient = 'index').transpose()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    

    

    
    

    
                
    #print final results
    

kmeans('databoth.csv')
    
    
