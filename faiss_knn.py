#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:27:35 2019

@author: shankarpm
"""
import numpy as np 
#from collections import Counter
from scipy import stats

class FaissKNNImpl:
    
    def __init__(self,k,faiss):
        self.k = k # k nearest neighbor value
        self.faissIns = faiss # FAISS instance
        self.index = 0
        self.gpu_index_flat = 0 
        self.train_labels = []  
        self.test_label_faiss_output = [] 
        
    def fitModel(self,train_features,train_labels): 
        self.train_labels = train_labels
        self.index = self.faissIns.IndexFlatL2(train_features.shape[1])   # build the index 
        self.index.add(train_features)       # add vectors to the index
        
    def fitModel_GPU(self,train_features,train_labels):
        no_of_gpus = self.faissIns.get_num_gpus()
        self.train_labels = train_labels
        self.gpu_index_flat = self.index = self.faissIns.IndexFlatL2(train_features.shape[1])   # build the index 
        if no_of_gpus > 0:
            self.gpu_index_flat = self.faissIns.index_cpu_to_all_gpus(self.index) 
            
        self.gpu_index_flat.add(train_features)       # add vectors to the index 
        return no_of_gpus
        
    def predict(self,test_features): 
        distance, test_features_faiss_Index = self.index.search(test_features, self.k) 
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index],axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        #for test_index in range(0,test_features.shape[0]):
        #    self.test_label_faiss_output[test_index] = stats.mode(self.train_labels[test_features_faiss_Index[test_index]])[0][0] #Counter(self.train_labels[test_features_faiss_Index[test_index]]).most_common(1)[0][0] 
        return self.test_label_faiss_output
    
    def predict_GPU(self,test_features): 
        distance, test_features_faiss_Index = self.gpu_index_flat.search(test_features, self.k) 
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index],axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        return self.test_label_faiss_output
      
    def getAccuracy(self,test_labels):
        accuracy = (self.test_label_faiss_output == test_labels).mean() 
        return round(accuracy,2) 
