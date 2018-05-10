#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:33:07 2018

@author: lorenzo
"""
import sys
import numpy as np
import random
import time
from sklearn.cluster import KMeans

MAX_FLOAT = sys.float_info.max


    
class Solution:
    #Private
    
    def __delete_assignment(self):
        self.__assignment = None
        
    #Public
               
    def __init__(self,assignment,cost = 0.0,alpha = 0.0):     
        self.__assignment = assignment 
        self.__cost = cost
        self.__alpha = alpha
        
    
    def getCost(self): 
        return self.__cost
    
    def getCostFromScratch(self, dataset, m):
        cost = 0
        centroid = self.assignmentToCentroid(dataset, m)
        for i in range(0, dataset.n):
                cost = cost + self.pcDist(i, centroid[self.__assignment[i]], dataset)
        return cost 

    def getAlpha(self):
        return self.__alpha
    
    def pcDist(self, p, center, dataset):
        total = 0
        for i in range(0,dataset.d):
            diff = dataset.data[p*(dataset.d)+i] - center[i]
            total = total + diff*diff
        return total
    
    def assignmentToCentroid(self, dataset, m):
        d = dataset.d 
        sizes = [None] * m
        centroid = [None] * m 
        
        for i in range(0,m):
            centroid[i] = [None] * d
            for j in range(0,d):
                centroid[i][j] = 0.0
            sizes[i] = 0
         
        for i in range(0,dataset.n):
            sizes[self.__assignment[i]] = sizes[self.__assignment[i]] + 1
            for j in range(0,dataset.d):
                centroid[self.__assignment[i]][j] = centroid[self.__assignment[i]][j] + dataset.data[i*d+j]
                
        for i in range(0,m):
            for j in range(0,d):
                if (sizes[i] != 0):
                    centroid[i][j] = (1.0*centroid[i][j])/sizes[i]
        sizes = None
        return centroid
    
    
    
    def centroidsToAssignment(self, centroid, dataset, m): 
        assignment = [None] * dataset.n
        for i in range(0, dataset.n):
            mindist = MAX_FLOAT
            for j in range(0,m):
                dist = self.pcDist(i, centroid[j], dataset)
                if np.any(dist < mindist):
                    mindist = dist
                    assignment[i] = j
        return assignment
    
    
    
class PbRun:

    
    def __init__(self, solution = [], time = 0.0, lastImprovment = 0, nbIter = 0.0, avgAlpha = 0.0, isValid = False):
        
        self.__solution = solution
        self.__time = time
        self.__lastImprovment = lastImprovment
        self.__nbIter = nbIter
        self.__avgAlpha = avgAlpha
        self.__isValid = isValid
        
    def getSolution(self):
        return self.__solution
    
    def getTime(self):
        return self.__time
    
    def getLastImprovment(self):
        return self.__lastImprovment
    
    def getNbIter(self):
        return self.nbIte
    
    def getAvgAlpha(self):
        return self.__avgAlpha
    
    def getIsValid(self):
        return self.__isValid
    
class Dataset:
    
    def __init__(self, dimensions, data, points):
        self.d = dimensions
        self.data = data
        self.n = points  

    
    
def readData():
        data = np.loadtxt("fisher.txt",skiprows=1)
        print(data)
        
        with open("fisher.txt") as f:
            first_line = f.readline()
        points, dimensions = map(int, first_line.split(" "))
        vetorized_data = [None] * (points * dimensions)
        
        for i in range(0,points):
            for j in range(0,dimensions):
                vetorized_data[i*dimensions+j] = data[i][j]
                
        print(vetorized_data)
        
def generateRandom(points,dimensions):
    
    data = [None] * points * dimensions
    for i in range(0,points):
        for j in range(0, dimensions):
            data[i*dimensions+j] = random.randint(0,1000)
    return data
        
        
def main():
    clusters = random.randint(2,10)
    points = random.randint(20,500)
    dimensions = random.randint(1,20)
    data = generateRandom(points,dimensions)
    print(points, dimensions, clusters)
    
    split_data = np.split(np.array(data),points)
    dataset = Dataset(dimensions, data, points)
    
    start_time = time.time()
    kmeans_results = KMeans(n_clusters = clusters, random_state=0).fit(split_data)
    total_time = time.time() - start_time
    
    kmeans_solution = kmeans_results.labels_
    solution = Solution(kmeans_solution,0,0)
    kmeans_clusters = kmeans_results.cluster_centers_
    kmeans_score = kmeans_results.score(split_data)
    pbRun = PbRun(solution,total_time)
    #print(kmeans_score)
    print(solution.getCostFromScratch(dataset,clusters))
    print(total_time)
    
   
    
    
main()
        
    
           



    
    
    

