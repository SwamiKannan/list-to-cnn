#!/usr/bin/env python
# coding: utf-8

from list_to_model import CNN_list
import torch.nn as nn

#Data structure (You can update the parameters or add additional stacks by copy pasting this structure)
conv=conv_list=[
            [
                [{'Conv':[{'out_channels':64,'kernel_size':3,'stride':1,'padding':1},2]}, 
                 {'Max_Pool':[[2,1,1],1]}   
                ],
                1                
            ],
            [
                [{'Conv':[[128,3,1,1],2]}, 
                 {'Max_Pool':[{'kernel_size':2,'stride':1,'padding':1},1]} 
                    
                ],
                1                
            ],
            [
                [{'Conv':[[256,3,1,1],3]}, 
                 {'Max_Pool':[[2,1,1],1]}  
                ]  
                , 1               
            ],
            [
                [{'Conv':[{'out_channels':512,'kernel_size':3,'stride':1,'padding':1},3]}, 
                 {'Max_Pool':[{'kernel_size':2,'stride':1,'padding':1},1]}   
                ]  
                , 2                 
            ]
        ]
dense_list=[
            [
                [{'Dense':[{'out_features':4096},2]}]  
                , 1                 
            ],
            [
                [{'Dense':[[1000],1]}] 

                , 1                 
            ]
        ]

#Code Execution
cnn=CNN_list()
cnn.build_conv_model(conv_list,55)
cnn.build_dense_model(dense_list,10)
cnn_model=cnn.model
print('List of layers in the model')
for param in cnn_model:
    print(param)
print('\n')
model_final=nn.Sequential(*cnn_model)
print('Final model')
print(model_final)





