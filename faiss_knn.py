#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:27:35 2019

@author: shankarpm
"""
import numpy as np 

class FaissKNNImpl:
    
    def __init__(self,k,faiss):
        self.k = k # k nearest neighbor value
        self.faissIns = faiss # FAISS instance
        self.index = 0
        self.distance = []
        self.db_search_index = []
        self.train_labels = []
        self.train_features = []
        self.test_label_faiss_output = []
        self.accuracy = 0
        
    def fitModel(self,train_features,train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.index = self.faissIns.IndexFlatL2(train_features.shape[1])   # build the index 
        self.index.add(train_features)       # add vectors to the index
        
    def predict(self,test_features,test_labels): 
        self.distance, self.db_search_index = self.index.search(test_features, self.k)
        k_threshold_check = round(self.k/2) #to check if we get 50% of the neighbors count match with the test data
        k_count = 0 # index for comparing with the threshold
        prediction_count = 0
        
        self.test_label_faiss_output = np.zeros(test_features.shape[0])
        for test_features_index in range(0,test_features.shape[0]):
            found_it = not_found_it = k_count = 0
            test_label_value = test_labels[test_features_index] 
            for k_index in range(0,self.k):    
              #test_index = 0 k_index = 0
              #train_labels[test_features_faiss_Index[test_index,k_index]]
              if(self.train_labels[self.db_search_index[test_features_index,k_index]] == test_label_value):
                 k_count = k_count + 1
                 found_it = test_label_value 
              else:
                 not_found_it = self.train_labels[self.db_search_index[test_features_index,k_index]] 
             
            if(k_count > k_threshold_check): #if the match is greater than the threshold then its a match.
                prediction_count = prediction_count + 1 
                self.test_label_faiss_output[test_features_index] = found_it
            else:
                self.test_label_faiss_output[test_features_index] = not_found_it
                    
        self.accuracy = ((prediction_count) / test_features.shape[0]) * 100
        return self.test_label_faiss_output
      
    def getAccuracy(self):
       return self.accuracy
