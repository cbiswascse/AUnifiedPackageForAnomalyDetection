# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:36:48 2021

@author: ChandrimaBiswas
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import hdbscan
import tracemalloc
from sklearn.metrics import ConfusionMatrixDisplay

clear = lambda:os.system('clear')


# extract data from csv files 
# seperate the Catagorical data and Numaric Data.
# Handel the missing value from data set. 
def extractData():
    
    DP = input("Please, Input the Location of CSV:")
  
    while True:
        
        DS = pd.read_csv(DP,low_memory=False)
        missingId = []
        missingValue = str(DS.isnull().values.any())
        if missingValue == "True":
            print("There has some missing value.")
            DS = DS.replace('Infinity', np.nan) 
            totalMissing = DS.isnull().sum().sum()
            percentMissing = (totalMissing / (DS.count().sum() + DS.isnull().sum().sum())) * 100
            for rows in DS: 
                if DS[rows].isnull().sum() != 0:
                   missingId.append(rows)
                   percentMissingRow=(DS[rows].isnull().sum()/DS[rows].count().sum())* 100
                   if percentMissingRow >= 30:
                       DS = DS.drop(rows, 1)  
            print("The missing values are in the columns:",missingId)
            print("Total number of missing Values:" , totalMissing)
            print(percentMissing,"%")
        
        num_cols = DS.columns.get_indexer(DS._get_numeric_data().columns)
        total_cols=DS.columns.get_indexer(DS.columns)
        while True:
             
             withCat = input("Do you want to include Catagorical data [y/n]:")
             
             
             if withCat == "y" or withCat == "n":
                 break
             else:   
                 print("Give your answer only in y or n .\n\n")   
        if withCat == "y":        
            Xdata = DS.iloc[:,total_cols[:-1]].values
            transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
            Xdata = transform.fit_transform(Xdata)
            Ydata = DS.iloc[:,total_cols[-1]].values
            le = preprocessing.LabelEncoder()
            le.fit(Ydata)
            Ydata=le.transform(Ydata)
            
        elif withCat == "n":
            Xdata = DS.iloc[:,num_cols[:-1]].values
            Ydata = DS.iloc[:,num_cols[-1]].values

        while True:
            
            decision = input("Scaling data with MinMaxScaler [y/n]:")
            
            if decision == "y" or  decision == "n":
                break
            else:
                
                print("Give your answer only in y or n.\n\n")
    
        if decision == "y": 
            Xdata =  MinMaxScaler(feature_range=(0, 1)).fit_transform(Xdata)
            return Xdata,Ydata
        
        else:
            return Xdata,Ydata

#Hierarchical Clustering
def hierarchicalClustering(data,Ydata): 
    from sklearn.cluster import AgglomerativeClustering
              
    while True:
        print("Agglomerative Clustering")              
        clusterNumber = input("How many clusters you want?:")       
        try:
            clusterNumber = int(clusterNumber)            
        except ValueError:           
            print("Error\n\n")
            
        if type(clusterNumber) == int:
            n = 0
            clusters = []          
            while n < clusterNumber:#Converting nClusters into an array of n clusters [n] for use it later
                clusters.append(n)
                n+=1
            break
        
    while True:
        linkage = input("The linkage criterion determines which distance to use [‘ward’, ‘complete’, ‘average’, ‘single’]:")
        
        if linkage == "ward" or linkage == "complete"or linkage == "average"or linkage == "single":
            break
        else:
            print("Give your answer Correctly.\n\n")

    print("\nClustering...\n")
    outliers_fraction = 0.1
    start_time = time.time()
    Aggo = AgglomerativeClustering(n_clusters = clusterNumber,affinity='euclidean',linkage=linkage,compute_distances=True)    
    print("Data Successfully Clustered By Agglomerative Clustering")
    AggoData = Aggo.fit(data)
    Xdata_Aggo = Aggo.fit_predict(data)
    Zdata = AggoData.labels_
    runTime=(time.time() - start_time)
    distance = Aggo.distances_
    number_of_outliers = int(outliers_fraction*len(distance))
    sorted_index_array = np.argsort(distance)
    sorted_array = distance[sorted_index_array]
    rslt = sorted_array[-number_of_outliers : ]
    threshold = rslt.min() 
    anomaly = (distance >= threshold).astype(int)
    HR = pd.crosstab(Ydata,Zdata)
    maxVal = HR.idxmax()    
    return Zdata,clusters,Xdata_Aggo,clusterNumber,anomaly,runTime,maxVal    



#K-Means Clustering
def kmeansClustering(data,labels): 
    from sklearn.cluster import KMeans           
    while True:
        print("Kmeans Clustering")
        clusterNumber = input("How many clusters you want?:")       
        try:
            clusterNumber = int(clusterNumber)            
        except ValueError:            
            print("Error\n\n")          
        if type(clusterNumber) == int:
            n = 0
            clusters = []          
            while n < clusterNumber:
                clusters.append(n)
                n+=1
            break       
    outliers_fraction = 0.01
    start_time = time.time()
    KMEANS = KMeans(n_clusters = clusterNumber, init = "k-means++",max_iter = 300,n_init = 10,random_state = 0)    
    print("Data Successfully Clustered with K-means")
    kmeans = KMEANS.fit(data)
    Xdata_Kmeans = KMEANS.fit_predict(data)
    Zdata = kmeans.labels_
    runTime=(time.time() - start_time)
    inertia = KMEANS.inertia_
    distance = getDistanceByPoint(data, KMEANS)
    number_of_outliers = int(outliers_fraction*len(distance))
    threshold = distance.nlargest(number_of_outliers).min() 
    anomaly = (distance >= threshold).astype(int)
    ClusterCenter=kmeans.cluster_centers_
    #Kmeans Results
    kmeansR = pd.crosstab(labels,Zdata)
    maxVal = kmeansR.idxmax()
    
    return Zdata,clusters,Xdata_Kmeans,ClusterCenter,inertia,clusterNumber,anomaly,runTime,maxVal

# return Series of distance between each point and his distance with the closest centroid
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = data[i]
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance

#HDBSCAN Algorithm
def hdbscanClustering(Xdata,Ydata):
    print("HDBSCAN Clustering")    
    #Computing DBSCAN
    clusterNumber = input("Minimun size of clusters you want?:")       
    try:
        clusterNumber = int(clusterNumber)            
    except ValueError:            
        print("Error\n\n")  
    start_time = time.time() 
    hdb = hdbscan.HDBSCAN(min_cluster_size=clusterNumber)  
    print("Data Successfully Clustered with HDBSCAN")    
    Zdata = hdb.fit_predict(Xdata)+1
    runTime = (time.time() - start_time)
    threshold = pd.Series(hdb.outlier_scores_).quantile(0.9)
    outliers = np.where(hdb.outlier_scores_ > threshold)[0]
    n_clusters = len(set(Zdata))
    n = 0  
    clusters = []
    while n  < n_clusters:
        clusters.append(n)
        n += 1
    HDBR = pd.crosstab(Ydata,Zdata)
    maxVal = HDBR.idxmax()
    return Zdata,clusters,outliers,runTime,maxVal


#DBSCAN Algorithm
def dbscanClustering(Xdata,Ydata):
    from sklearn.cluster import DBSCAN
    
    while True:
        print("DBSCAN Clustering")
        epsilon = input("epsilon in Decimal:")        
        try:
            epsilon = float(epsilon)           
        except ValueError:            
            print("Decimal numbers only")
            
            
        if type(epsilon) == float:
            break
        
    while True:
        minSamples = input("Min Samples In Integer:")       
        try:
            minSamples = int(minSamples)            
        except ValueError:
            print("Integer Numbers only")            
        if type(minSamples) == int:
            break
        
    

    #Computing DBSCAN
    start_time = time.time() 
    db = DBSCAN(eps= epsilon, min_samples = minSamples,algorithm ='auto').fit(Xdata)   
    print("Data Successfully Clustered by DBSCAN")
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True    
    Zdata = db.labels_
    runTime=(time.time() - start_time)
    nClusters = len(set(Zdata))
    noise_ = list(Zdata).count(-1)
    clusters = []
    if noise_ > 0:
        i = -1  
        while i + 1 < nClusters:
            clusters.append(i)
            i += 1
    else:
        i=0
        while i < nClusters:
            clusters.append(i)
            i += 1     
    #DBSCAN Results
    dbscanR = pd.crosstab(Ydata,Zdata)
    maxVal = dbscanR.idxmax()
    
    return Zdata,clusters,noise_,runTime,maxVal







def isolationForest(Xdata,Ydata):#Isolation Forest algorithm
    from sklearn.ensemble import IsolationForest
    
    print("Isolation Forest Clustering")
    while True:
        contamination = input("Contamination value between [0,0.5]: ")
        
        try:
            contamination = float(contamination)           
        except ValueError:           
            print("Enter a Number between [0,0.5]")           
        if type(contamination) == float and (contamination >= 0 and contamination <= 0.5):
            break   
    start_time = time.time() 
    Zdata = IsolationForest(max_samples = "auto",contamination = contamination).fit_predict(Xdata)
    print("Data Successfully Clustered by Isolation Forest")
    runTime=(time.time() - start_time)    
    Zdata = np.array(Zdata,dtype = object)   
    n = -1  
    clusters = []        
    IFR = pd.crosstab(Ydata,Zdata)
    maxVal = IFR.idxmax()
    
    while n < len(IFR.columns):
        clusters.append(n)
        n += 2
        
    return Zdata,clusters,runTime,maxVal


    

def LFO(Xdata,Ydata):#Local Outlier Factor algorithm
    from sklearn.neighbors import LocalOutlierFactor 
    
    print("Local Outlier Factor Clustering")
    while True:
        contamination = input("Contamination value between [0,0.5]: ")        
        try:
            contamination = float(contamination)
            
        except ValueError:
            
            print("Enter a Number")
            
        if type(contamination) == float and (contamination > 0 and contamination <= 0.5):
            break
        
    start_time = time.time() 
    lof = LocalOutlierFactor(contamination = contamination,algorithm = 'auto').fit_predict(Xdata)
    print("Data Successfully Clustered by Local Outlier Factor")
    runTime=(time.time() - start_time)    
    n = -1  
    clusters = []   
    LOFR = pd.crosstab(Ydata,lof)
    maxVal = LOFR.idxmax()
    while n < len(LOFR.columns):
        clusters.append(n)
        n += 2   
    return lof,clusters,runTime,maxVal

def hierarchicalVisualization(Xdata,Xdata_aggo,nClusters,anomaly): 
    # Visualising the clusters
    length=len(Xdata)
    color=['yellow','blue', 'green','cyan','magenta','violet', 'Antique ruby','Aqua','Blush'  ]
    for i in range (0,nClusters):
        plt.scatter(Xdata[Xdata_aggo == i, 0], Xdata[Xdata_aggo == i,1],s = length, c=color[i], label = 'Cluster'+ str(i+1))
    plt.scatter(Xdata[:-1][anomaly == 1,0], Xdata[:-1][anomaly == 1,1], s = length, c = 'red', label = 'Outliers')
    plt.title('Agglomerative Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(anomaly[anomaly==1]),len(anomaly[anomaly==0])])
    plt.xticks(Xdata_pos, bars)
    barlist[0].set_color('r')
    plt.title('Agglomerative Bar Chart')
    plt.show()
    
def hdbscanVisualization(Xdata,outliers): 
    # Visualising the clusters
    plt.scatter(Xdata.T[0],Xdata.T[1], s=100, linewidth=0, c='gray', alpha=0.25)
    plt.scatter(Xdata[outliers].T[0], Xdata[outliers].T[1],s=100, linewidth=0, c='red', alpha=0.5)
    plt.title('HDBSCAN Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(Xdata[outliers].T[0]),len(Xdata.T[0])])
    plt.xticks(Xdata_pos, bars)
    plt.title('HDBSCAN Bar Chart')
    barlist[0].set_color('r')
    plt.show()    


def kmeansVisualization(Xdata,Xdata_Kmeans,ClusterCenter,nClusters,anomaly): 
    # Visualising the clusters
    color=['yellow','blue', 'green','cyan','magenta','violet', 'ruby','Aqua','Blush'  ]
    for i in range (0,nClusters):
        plt.scatter(Xdata[Xdata_Kmeans == i, 0], Xdata[Xdata_Kmeans == i,1],s = 100, c=color[i], label = 'Cluster'+ str(i+1))
    plt.scatter(ClusterCenter[:,0], ClusterCenter[:,1], s = 300, c = 'Black', label = 'Centroids')
    plt.scatter(Xdata[anomaly == 1,0], Xdata[anomaly == 1,1], s = 100, c = 'red', label = 'Outliers')
    plt.title('K-means Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(anomaly[anomaly==1]),len(anomaly[anomaly==0])])
    plt.xticks(Xdata_pos, bars)
    plt.title('K-means Bar Chart')
    barlist[0].set_color('r')
    plt.show()
    
def dbscanVisualization(Xdata,dblabels,dbClusters): 
    # Visualising the clusters
    color=['yellow','blue', 'green','cyan','magenta','violet', 'orange']
    a=0
    if dbClusters[a] == -1:
        plt.scatter(Xdata[dblabels == dbClusters[0], 0], Xdata[dblabels == dbClusters[0],1],s = 100, c='red', label = 'Cluster'+ str(1))
        a=+1
    for i in range (a,len(dbClusters)):
        plt.scatter(Xdata[dblabels == dbClusters[i], 0], Xdata[dblabels == dbClusters[i],1],s = 100, c=color[i % len(color)], label = 'Cluster'+ str(i+1))
    plt.title('DBSCAN Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(Xdata[dblabels==-1]),len(Xdata[dblabels!=-1])])
    plt.xticks(Xdata_pos, bars)
    plt.title('DBSCAN Bar Chart')
    barlist[0].set_color('r')
    plt.show()
    
def isolationForestVisualization(Xdata,ifLabels,ifNclusters): 
    # Visualising the clusters
    color=['red', 'green']
    for i in range (0,len(ifNclusters)):
        plt.scatter(Xdata[ifLabels == ifNclusters[i], 0], Xdata[ifLabels == ifNclusters[i],1],s = 100, c=color[i], label = 'Cluster'+ str(i+1))
    plt.title('Isolation Forest Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(Xdata[ifLabels==-1]),len(Xdata[ifLabels==1])])
    plt.xticks(Xdata_pos, bars)
    plt.title('Isolation Forest Bar Chart')
    barlist[0].set_color('r')
    plt.show()
    
def LfoVisualization(Xdata,LOFlabels,lofClusters): 
    # Visualising the clusters
    color=['red', 'green']
    for i in range (0,len(lofClusters)):
        plt.scatter(Xdata[LOFlabels == lofClusters[i], 0], Xdata[LOFlabels == lofClusters[i],1],s = 100, c=color[i], label = 'Cluster'+ str(i+1))
    plt.title('Local Factor Outlier Clustering')
    plt.show()
    bars = ('anomaly','normal')
    Xdata_pos = range(len(bars))
    # Create bars
    barlist=plt.bar(Xdata_pos, [len(Xdata[LOFlabels==-1]),len(Xdata[LOFlabels==1])])
    plt.xticks(Xdata_pos, bars)
    plt.title('Local Factor Outlier Bar Chart')
    barlist[0].set_color('r')
    plt.show()
    
def kpm(Zdata,Ydata,maxValue,clusters):#Performance Metrics Score for Kmeans
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix

    
    
    number = 0 
    clusterDictionary  = {}  
    f1Score = 0 
    method = ''
    
    while number < len(clusters):
        clusterDictionary[clusters[number]] = maxValue[number] 
        number+=1
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
            
    Ydata = np.array(Ydata,dtype = int) 
    
    while True:
        
        method = input("Average Method[weighted,micro,macro,binary]:")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
    #score metric   
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    return f1Score, precision, recall, clusterDictionary,confusionMatrix

def dbpm(Zdata,Ydata,clusters,maxValue):#Performance Metrics score for DBSCAN
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    
    number = 0 
    i = -1 
    clusterDictionary  = {}  
    f1Score = 0
    method = ''
    
    while number < len(clusters):
        clusterDictionary[clusters[number]] = maxValue[i] 
        number+=1
        i+=1
    
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
    
    Ydata = np.array(Ydata,dtype = int) 
    while True:
        
        method = input("Select one of Average Method (weighted,micro,macro,binary):")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
        
        else:
            
            print("Error\n\n")
    #score metric
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    
    return f1Score,precision, recall, clusterDictionary, confusionMatrix

def IFpm(Zdata,Ydata,clusters,maxValue): #Performance Metrics score for IF
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    
    number = 0 
    i = -1
    f1Score = 0
    clusterDictionary  = {} 
    
    while number < len(clusters): 
        clusterDictionary[clusters[number]] = maxValue[i] 
        number+=1
        i+=2
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
    Ydata = np.array(Ydata,dtype = int)
    Zdata = np.array(Zdata,dtype = int)
    while True:
        
        method = input("Select one of Average Method (weighted,micro,macro,binary):")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
        
        else:
            
            print("Error\n\n")
    #score metric
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    
    return f1Score,precision, recall, clusterDictionary, confusionMatrix

def LFOpm(Zdata,Ydata,clusters,maxValue): #Performance Metrics score for local outlier factor
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    
    number = 0 
    i = -1 
    f1Score = 0
    clusterDictionary  = {} 
    
    while number < len(clusters): 
        clusterDictionary[clusters[number]] = maxValue[i] 
        number+=1
        i+=2
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
    Ydata = np.array(Ydata,dtype = int)
    Zdata = np.array(Zdata,dtype = int)
    while True:
        
        method = input("Select one of Average Method (weighted,micro,macro,binary)::")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
        
        else:
            
            print("Error\n\n")
    #score metric
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    
    return f1Score,precision, recall, clusterDictionary, confusionMatrix

def HDBpm(Zdata,Ydata,clusters,maxValue): #Performance Metrics score for HDBSCAN
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    
    number = 0 
    i = 0 
    f1Score = 0
    clusterDictionary  = {}  
    
    while number < len(clusters): 
        clusterDictionary[clusters[number]] = maxValue[i] 
        number+=1
        i+=1
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
    Ydata = np.array(Ydata,dtype = int)
    Zdata = np.array(Zdata,dtype = int)
    while True:
        
        method = input("Select one of Average Method (weighted,micro,macro,binary)::")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
        
        else:
            
            print("Error\n\n")
    #score metric
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    
    return f1Score,precision, recall, clusterDictionary, confusionMatrix

def AGGLOpm(Zdata,Ydata,clusters,maxValue): #Performance Metrics score for Agglomerative
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    
    number = 0 
    i = 0 
    f1Score = 0
    clusterDictionary  = {} 
    
    while number < len(clusters): 
        clusterDictionary[clusters[number]] = maxValue[i] 
        number+=1
        i+=1
        
    Zdata[:] = [clusterDictionary[item] for item in Zdata[:]] 
    Ydata = np.array(Ydata,dtype = int)
    Zdata = np.array(Zdata,dtype = int)
    while True:
        
        method = input("Select one of Average Method (weighted,micro,macro,binary)::")
        
        if method == "weighted" or method == "micro" or method == "macro" or method == 'binary':
            break
        
        else:
            
            print("Error\n\n")
    f1Score = f1_score(Ydata,Zdata, average = method)
    precision = precision_score(Ydata,Zdata, average = method)
    recall =recall_score(Ydata,Zdata, average = method)
    confusionMatrix=confusion_matrix(Ydata,Zdata)
    
    return f1Score,precision, recall, clusterDictionary,confusionMatrix


    
def ClusterView():
    clear()
    print("Developed By Chandrima Biswas")
    data,labels = extractData()
    print("Data set is extrcted successfully.")
    runTimeall=[]
    outLier=[]
    F1scoer=[]
    precision=[]
    recall=[]
    algo=[]
    total_mem_usage=[]
    while True:
        while True:
            print("Available Clusering Algorithm\n\n Kmeans\n Dbscan \n Isolation Forest \n Local Factor Outlier \n Hdbscan \n Agglomerative")
            clustering = input("Choose your Algorithm:")
                
            if clustering == "Kmeans" or clustering == "Dbscan" or clustering == "Isolation Forest" or clustering == "Local Factor Outlier" or clustering == "Hdbscan"or clustering == "Agglomerative":
                break
            else:
                print("Please Check the algorithom name.\n\n")
        
            
        if clustering == "Kmeans":
            # starting the monitoring
            tracemalloc.start()
            kmeanLabels,kmeanClusters,Xdata_Kmeans,kmeanClusterCenter,kmeanInertia,numberClusters,anomaly,runTime,maxKmeanValue = kmeansClustering(data,labels)
            # displaying the memory
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            #mem_usage = memory_usage(kmeansClustering)
            print("Anomaly Ditection: ", data[anomaly==1])
            #print('Memory usage : %s' % mem_usage)
            outlierPercent=(len(data[anomaly==1])/len(data))*100
            outLier.append(outlierPercent)
            kmeansVisualization(data,Xdata_Kmeans,kmeanClusterCenter,numberClusters,anomaly)
            kmeansF1,kmeansP,kmeansR,clusterAssigned, kConMtx = kpm(kmeanLabels,labels,maxKmeanValue,kmeanClusters)
            F1scoer.append(kmeansF1)
            precision.append(kmeansP)
            recall.append(kmeansR)
            algo.append(clustering)
            cm_display = ConfusionMatrixDisplay(kConMtx).plot()
            plt.title('K-means Confusion Matrix')
            plt.show()
            
            
                
        elif clustering == "Dbscan":
            # starting the monitoring
            tracemalloc.start()
            dbscanLabels,dbscanClusters,numberNoises,runTime,maxDbscanValue = dbscanClustering(data,labels)
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            dbscanVisualization(data,dbscanLabels,dbscanClusters)
            outlierPercent=((numberNoises)/len(data))*100
            outLier.append(outlierPercent)
            dbscanF1,dbscanP,dbscanR,clusterAssigned, dConMtx = dbpm(dbscanLabels,labels,dbscanClusters,maxDbscanValue)
            F1scoer.append(dbscanF1)
            precision.append(dbscanP)
            recall.append(dbscanR)
            algo.append(clustering)
            cm_display = ConfusionMatrixDisplay(dConMtx).plot()
            plt.title('DBSCAN Confusion Matrix')
            plt.show()
            
            
            
        elif clustering == "Isolation Forest":
            # starting the monitoring
            tracemalloc.start()
            isofLabels,isofNclusters,runTime,maxIsofValue = isolationForest(data,labels)
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            isolationForestVisualization(data,isofLabels,isofNclusters)
            outlierPercent=(len(data[isofLabels==-1])/len(data))*100
            outLier.append(outlierPercent)
            isofF1,isofP,isofR,clusterAssigned,iConMtx = IFpm(isofLabels,labels,isofNclusters,maxIsofValue)
            F1scoer.append(isofF1)
            precision.append(isofP)
            recall.append(isofR)
            algo.append('I F')
            cm_display = ConfusionMatrixDisplay(iConMtx).plot()
            plt.title('Isolation Forest Confusion Matrix')
            plt.show()
            
    
    
            
        elif clustering == "Local Factor Outlier":
            # starting the monitoring
            tracemalloc.start()
            LfoLabels,lfoClusters,runTime,maxLfoValue = LFO(data,labels)
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            LfoVisualization(data,LfoLabels,lfoClusters)
            outlierPercent=(len(data[LfoLabels==-1])/len(data))*100
            outLier.append(outlierPercent)
            lfoF1,lfoP,lfoR,clusterAssigned, lConMtx = LFOpm(LfoLabels,labels,lfoClusters,maxLfoValue)
            F1scoer.append(lfoF1)
            precision.append(lfoP)
            recall.append(lfoR)
            algo.append('LFO')
            cm_display = ConfusionMatrixDisplay(lConMtx).plot()
            plt.title('Local Factor Outlier Confusion Matrix')
            plt.show()
            
            
            
        elif clustering == "Hdbscan":
            # starting the monitoring
            tracemalloc.start()
            hdblabels,hdbClusters,outliers,runTime,maxHDBvalue = hdbscanClustering(data,labels)
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            hdbscanVisualization(data,outliers)
            outlierPercent=(len(data[outliers].T[0])/len(data))*100
            outLier.append(outlierPercent)
            hdbscanF1,hdbscanP,hdbscanR,clusterAssigned, hdConMtx  = HDBpm(hdblabels,labels,hdbClusters,maxHDBvalue)
            F1scoer.append(hdbscanF1)
            precision.append(hdbscanP)
            recall.append(hdbscanR)
            algo.append(clustering)
            cm_display = ConfusionMatrixDisplay(hdConMtx).plot()
            plt.title('Hdbscan Confusion Matrix')
            plt.show()
            
            
        
        elif clustering == "Hierarchical":
            # starting the monitoring
            tracemalloc.start()
            agglolabels, aggloclusters,Xdata_Aggo, aggloClusterNumber,anomaly,runTime,maxAggloValue=hierarchicalClustering(data,labels)
            memory=tracemalloc.get_traced_memory()
            mem_usage=memory[1]-memory[0]
            print('Memory Used:',mem_usage)
            print('runTime:',runTime)
            # stopping the library
            tracemalloc.stop()
            hierarchicalVisualization(data,Xdata_Aggo, aggloClusterNumber,anomaly)
            outlierPercent=(len(anomaly[anomaly==1])/len(data))*100
            outLier.append(outlierPercent)
            aggloF1,aggloP,aggloR,clusterAssigned, aggloConMtx = AGGLOpm(agglolabels,labels,aggloclusters,maxAggloValue)
            F1scoer.append(aggloF1)
            precision.append(aggloP)
            recall.append(aggloR)
            algo.append('Agglomerative')
            cm_display = ConfusionMatrixDisplay(aggloConMtx).plot()
            plt.title('Agglomerative Confusion Matrix')
            plt.show()
            
            
        total_mem_usage.append(mem_usage)
        runTimeall.append(runTime)            
        while True: 
            
            decision = input("Do you want to try another Clustering Algorithm (y/n)?:")
            if decision == "y" or  decision == "n":
                break
            else:
                
                print("Error\n\n")
                
        if decision == "y":

            action = input("Are you Going to use New data (y/n)?:")
            if action == "y":  
                data = extractData()
                print("Data set is extrcted successfully.")
                
            elif action == "n":
                data=data
              
        elif decision == "n":
            if len(algo) >1: 
                plt.bar(np.arange(len(runTimeall)), runTimeall,width=.9)
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10)
                plt.title('Run Time for Algorithms')                
                plt.show()
                plt.bar(np.arange(len(total_mem_usage)), total_mem_usage,width=.9,color="#AA6644")
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10)
                plt.title('Memory used in Algorithms')                
                plt.show()
                plt.bar(np.arange(len(algo)), outLier ,width=.9,color='red')
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10)
                plt.title('% of detected Anomalies')
                plt.show()
                plt.bar(np.arange(len(F1scoer)), F1scoer,width=.9,color="#6D6D84")
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10) 
                plt.title('F1 Scoring values of Algorithms')
                plt.show()
                plt.bar(np.arange(len(precision)), precision,width=.9,color="#134F5C")
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10)
                plt.title('Precision values of Algorithms')                
                plt.show()
                plt.bar(np.arange(len(recall)), recall,width=.9,color="#444466")
                plt.xticks(np.arange(len(algo)), algo,rotation=10,size=10) 
                plt.title('Recall values of Algorithms')
                plt.show()
            break
        
        else:
            clear()

if __name__ == "__main__":
    # execute only if run as a script
    main()