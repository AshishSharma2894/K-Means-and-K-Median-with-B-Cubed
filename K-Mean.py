import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
The Program is to compile four different datasets and merge them into Single file which 
can further be used to find the K-Means and K-Medians.
Here further We are Calculating K-Means and K-Medians with B-Cubed Evalution with help of L2 Normalisation
which helps us into better understanding of which type of operation is better in performing overall more precise 
Clusters.
Have also calculated the Precision, Recall and F-Score for all the different type for better understanding of which 
function is performing the task in much better way.

"""
data_1= pd.read_csv("C:\library/animals.csv",sep='\s+',header=None)
data_2= pd.read_csv("C:\library/countries.csv",sep='\s+',header=None)
data_3= pd.read_csv("C:\library/fruits.csv",sep='\s+',header=None)
data_4= pd.read_csv("C:\library/veggies.csv",sep='\s+',header=None)

#print(data_1)

def DataCompiled():
    """The Purpose of this function is to compile all the data into one dataset which is further 
    helpful to workaround and also have classified animals, countries, fruits and veggies into 
    into different clases to further differentiate

    Returns:
        NDArray: Returns the complete dataset
    """
    data_1['class']='0'
    data_2['class']='1'
    data_3['class']='2'
    data_4['class']='3'

    frames=[data_1,data_2,data_3,data_4]
    dataset=np.array(pd.concat(frames))
    #print(dataset)
    return dataset

def distance(X,Y):
    #Return the Euclidean distance between X and Y
    return np.linalg.norm(X-Y)

def manDistance(X,Y):
    #Return the Manhattan distance between X and Y
    p1=np.array(X)
    p2=np.array(Y)
    return np.sum(np.abs(p1-p2))

def assign(centroids, dataset, clusters,k):
    """The Assign Function helps overall assigning the value to the functions used in the K-Means and K-Medians

    Args:
        centroids (NDArray(Float64)): This helps us store the value for the centroid of the cluster
        dataset (NDArray): Dataset which we are using in the program to calculate K-Means and K-Medians
        clusters (NDArray(Float64)): The different clusters formed by the program to classify them into different categories
        k (int): The number of times we have to iterate the function
    """
    numOfObjects = len(dataset)
    #for every object in the dataset
    for i in range(numOfObjects):
        X = dataset[i, 1:-1]
        #find the closest centroid
        centroidsOfX = -1
        distanceToClosestcentroids = np.Inf
        for y in range(k):
            currentcentroids = centroids[y,:]
            dist = distance(X, currentcentroids)
            if dist < distanceToClosestcentroids:
                #Finally found closer Centroid
                distanceToClosestcentroids = dist
                centroidsOfX = y
        #Assign to X its closest centroid
        clusters[i] = int(centroidsOfX)

def objectiveFunc(centroids,dataset,k):
    numOfObject=len(dataset)
    clusters= np.zeros((len(dataset),1))
    #Assign objects to the closest centroid
    assign(centroids,dataset,clusters,k)
    obj=0
    for i in range(numOfObject):
        obj=obj+distance(dataset[i,1:-1],centroids[int(clusters[i,:])])
    return obj,clusters

def objectiveFunc_M(centroids,dataset,k):
    numOfObject=len(dataset)
    clusters=np.zeros((len(dataset),1))
    #Assign objects to the closest centroid
    assign(centroids,dataset,clusters,k)
    #print(k)
    obj=0
    for i in range(numOfObject):
        #Calculate the L1 distance between object and its centroid
        obj= obj+distance(dataset[i,1:-1],centroids[int(clusters[i,:])])
    return obj,clusters

def KMeans(k,dataset,clusters,MaxIter=10):
    numOfObjects= len(dataset)
    np.random.seed(45)
    centroidsInd=np.random.choice(numOfObjects,k,replace=False)
    # Store centroids vectors in centroids
    centeroids=np.empty((0,300))
    for i in range(len(centroidsInd)):
        centeroids=np.vstack((centeroids,dataset[centroidsInd[i],1:-1]))
    assign(centeroids,dataset,clusters,k)
    #print(clusters)
    tempClusters= np.copy(clusters)
    bestObjective,tempClusters=objectiveFunc(centeroids,dataset,k)
    print("===========================\n""Initial objective function using",k,"clusters :%.2f"% bestObjective,"Initial centroids indices: ",centroidsInd)
    isObjectiveImproved= False
    counter=0

    for i in range(MaxIter):
        isObjectiveImproved=False
        #update the temp_centroids using mean
        temp_centroids=np.empty((0,300),int)
        for index in range(k):
            clusterDataset= np.empty((0,302),int)
            for x in range(numOfObjects):
                if int(tempClusters[x])==index:
                    clusterDataset= np.vstack((clusterDataset,dataset[x]))
                    #calculate the mean vector of a cluster
            new_centroid = np.mean(clusterDataset[:,1:-1],axis=0)
            temp_centroids= np.vstack((temp_centroids,new_centroid))
        tempObjective, tempClusters= objectiveFunc(temp_centroids,dataset,k)
        if tempObjective < bestObjective:
            isObjectiveImproved=True
        
        if isObjectiveImproved:
            centeroids=temp_centroids
            bestObjective=tempObjective
            counter+=1
        
        else:
            break
        print("Improved Objective Function Value after ",counter,"Times of iteration: %.2f" %bestObjective)
        return tempClusters




def kMedians(k, dataset, clusters, maxIter=10):
    numOfObjects = len(dataset)
    np.random.seed(45)
    centroidsInd = np.random.choice(numOfObjects, k, replace=False)
    # The Purpose here is to store the centroids under centroids
    centroids = np.empty((0,300))
    for i in range(len(centroidsInd)):
      centroids = np.vstack((centroids, dataset[centroidsInd[i],1:-1]))
    assign(centroids, dataset, clusters, k)
    #print(clusters)
    tempClusters = np.copy(clusters)
    bestObjective, tempClusters = objectiveFunc_M(centroids, dataset,k)
    print( "===========================\n""Initial objective function using",k,"clusters: %.2f" % bestObjective, "    Initial centroids indices: ", centroidsInd)
    isObjectiveImproved = False
    counter = 0

    for i in range(maxIter):
      isObjectiveImproved = False
      # update the temp_centroids using median
      temp_centroids = np.empty((0,300), int)
      for index in range(k):
        clusterDataset = np.empty((0,302), int)
        for x in range(numOfObjects):
          if int(tempClusters[x]) == index:
            clusterDataset = np.vstack((clusterDataset, dataset[x]))
            # find the median vector of a cluster
        new_centroid = np.median(clusterDataset[:,1:-1], axis = 0)
        temp_centroids = np.vstack((temp_centroids, new_centroid))

      tempObjective, tempClusters= objectiveFunc_M(temp_centroids, dataset, k)
      if tempObjective < bestObjective:
        isObjectiveImproved = True
      if isObjectiveImproved:
            centroids = temp_centroids
            bestObjective = tempObjective
            counter += 1
      else:
        break
    print("improved objective function value after ",counter,"times of iteration: %.2f" % bestObjective)
    return tempClusters

def B_CUBED(clusters, dataset, k):
  precision = 0
  recall = 0
  Final_Precision = 0
  Final_Recall = 0
  f_score = 0
  numOfObjects = len(dataset)
  Assigned_Cluster = np.empty((0, 4))
  for x in range(k):
      Animal_Class = 0
      Country_Class = 0
      Fruit_Class = 0
      Veggie_Class = 0
      cluster_set = np.empty((k, 4))
      for i in range(numOfObjects):
        if clusters[i] == x:
          if dataset[i, -1] == '0':
            Animal_Class += 1
          elif dataset[i, -1] == '1':
            Country_Class += 1
          elif dataset[i, -1] == '2':
            Fruit_Class += 1
          elif dataset[i, -1] == '3':
            Veggie_Class += 1
      cluster_set = [Animal_Class, Country_Class, Fruit_Class, Veggie_Class]
      Assigned_Cluster = np.vstack((Assigned_Cluster, cluster_set))
  for i in range(numOfObjects):
    precision = Assigned_Cluster[int(clusters[i]),int(dataset[i,-1])] / np.sum(Assigned_Cluster[int(clusters[i]),:])
    Final_Precision += precision
    recall = Assigned_Cluster[int(clusters[i]),int(dataset[i,-1])] / np.sum(Assigned_Cluster[:, int(dataset[i,-1])])
    Final_Recall += recall
    f_score += (2*recall*precision)/(recall+precision)
  print(Final_Precision/len(dataset))
  print(Final_Recall/len(dataset))
  print(f_score/len(dataset))
  return Final_Precision/len(dataset), Final_Recall/len(dataset),f_score/len(dataset)

def plot_BCUBED(precision, recall, f_score, title):
    figure = plt.figure(1, figsize = (12, 4))
    plt.title(title, fontsize=14)
    x = [1,2,3,4,5,6,7,8,9]
    plt.plot(x,precision,'o-',color = 'c',label='precision', alpha=0.6)
    plt.plot(x,recall,'v-',color = 'r',label='recall', alpha=0.6)
    plt.plot(x,f_score,'d-',color = 'g',label='F-score', alpha=0.6)
    plt.xlabel('Total Number of Clusters(k)')
    plt.legend(loc = 'best')
    plt.show()

def L2Normalization_Func(dataset):
    # for every object in dataset, l2 normalise the object and store it
    for i in range(len(dataset)):
        dataset[i,1:-1]=dataset[i,1:-1]/np.linalg.norm(dataset[i,1:-1])
        #print(dataset)
    return dataset


def Final_KMeans_Without_L2():
    title = 'K-Means algorithm without L2 B-CUBED Evaluation'
    dataset = DataCompiled()
    clusters = np.zeros((len(dataset), 1))
    precision = []
    recall = []
    f_score = []
    print("K-Means Algorithm B-CUBED Evaluation")
    for i in range(1, 10):
        clusters = KMeans(i, dataset, clusters)
        #print(clusters)
        prec, rcal, fscr = B_CUBED(clusters, dataset, i)
        precision.append(prec)
        recall.append(rcal)
        f_score.append(fscr)
    plot_BCUBED(precision, recall, f_score, title)



def Final_KMeans_L2():
    title = 'K-Means algorithm with L2 B-CUBED Evaluation'
    dataset = DataCompiled()
    dataset = L2Normalization_Func(dataset)
    clusters = np.zeros((len(dataset), 1))
    precision = []
    recall = []
    f_score = []
    print("K-Means algorithm with L2 Normalisation B-CUBED Evaluation")
    for i in range(1, 10):
        clusters = KMeans(i, dataset, clusters)
        prec, rcal, fscr = B_CUBED(clusters, dataset, i)
        precision.append(prec)
        recall.append(rcal)
        f_score.append(fscr)
    plot_BCUBED(precision, recall, f_score, title)

def Final_KMedians_Without_L2():
    title = 'K-Medians algorithm without L2 B-CUBED Evaluation'
    dataset = DataCompiled()
    clusters = np.zeros((len(dataset), 1))
    precision = []
    recall = []
    f_score = []
    print("K-Medians algorithm B-CUBED Evaluation")
    for i in range(1, 10):
        clusters = kMedians(i, dataset, clusters)
        prec, rcal, fscr = B_CUBED(clusters, dataset, i)
        precision.append(prec)
        recall.append(rcal)
        f_score.append(fscr)
    plot_BCUBED(precision, recall, f_score, title)


def Final_KMedians_L2():
    title = 'K-Medians algorithm with L2 B-CUBED Evaluation'
    dataset = DataCompiled()
    dataset = L2Normalization_Func(dataset)
    clusters = np.zeros((len(dataset), 1))
    precision = []
    recall = []
    f_score = []
    print("K-Medians algorithm with l2 normalisation B-CUBED Evaluation")
    for i in range(1, 10):
        clusters = kMedians(i, dataset, clusters)
        prec,recl,fscr = B_CUBED(clusters, dataset, i)
        precision.append(prec)
        recall.append(recl)
        f_score.append(fscr)
    plot_BCUBED(precision, recall, f_score, title)


def Final_Complication():
    Final_KMeans_Without_L2()
    Final_KMeans_L2()
    Final_KMedians_Without_L2()
    Final_KMedians_L2()


Final_Complication()
