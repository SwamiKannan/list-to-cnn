import torch.nn as nn
class CNN_list(nn.Module):
    '''
        We are going to create a nn.Module-based neural network model. Large architectures have a large number of components 
        e.g. VGG-16 has:
                1 stack of [2 3X3X64 conv layers followed by a MaxPool]
                1 stack of [2 3X3X128 conv layers followed by a MaxPool]
                1 stack of [3 3X3X256 conv layers followed by a MaxPool ]
                2 stacks of [3 3X3X512 conv layers followed by a MaxPool ]
                1 stack of [2 4096 Dense Layers]
                1 1000 Dense Layer
    '''
    def __init__(self): #image_height, image_width,img_channels to be added to see if Linear and Conv functions can be integrated
        super().__init__()
        self.model=[]

    def build_conv_model(self,layer_list,in_size):
        input_size=in_size
        initial=True #Flag set 
        for stack in layer_list:
            stack_layers, stack_repeats=stack[0],stack[1]
            for _ in range(stack_repeats):
                for sl in stack_layers:
                    algo , params, layer_repeats=list(sl.keys())[0], list(sl.values())[0][0],list(sl.values())[0][1]
                    first_list=True
                    for i in range(layer_repeats):
                        if algo=='Conv':
                            if isinstance(params, dict):
                                out_size=params['out_channels']
                                if initial:
                                    input_channels=in_size
                                    initial=False
                                params['in_channels']=input_channels
                                self.model.append(nn.Conv2d(**params))
                                self.model.append(nn.ReLU())
                                input_channels=out_size
                            elif isinstance(params, list):
                                if initial:
                                    input_channels=in_size
                                    initial=False
                                if first_list:
                                    params=[input_channels]+params
                                    first_list=False
                                else: 
                                    params[0]=input_channels                                       
                                out_size=params[1]
                                self.model.append(nn.Conv2d(*params))
                                input_channels=out_size
                            else:
                                print('Parameters neither passed as a dict or as an array')
                        elif algo=='Max_Pool':
                            if isinstance(params, dict):
                                self.model.append(nn.MaxPool2d(**params))
                            elif isinstance(params, list):
                                self.model.append(nn.MaxPool2d(*params))
                            else:
                                print('Parameters neither passed as a dict or as an array')
                        else:
                            print('The layer cannot be added to the model')
                            if initial==True:
                                initial=False

    def build_dense_model(self,layer_list,in_feat):
        input_size=in_feat
        initial=True
        for stack in layer_list:
            stack_layers, stack_repeats=stack[0],stack[1]
            for _ in range(stack_repeats):
                for sl in stack_layers:
                    algo , params, layer_repeats=list(sl.keys())[0], list(sl.values())[0][0],list(sl.values())[0][1]
                    first_list=True
                    for _ in range(layer_repeats):
                        if algo=='Dense':
                            if isinstance(params, dict):
                                out_size=params['out_features']
                                input_features=input_size if initial else input_features
                                params['in_features']=input_features
                                self.model.append(nn.Linear(**params))
                                self.model.append(nn.ReLU())
                                input_features=out_size
                            elif isinstance(params, list):
                                if first_list:
                                    params=[input_features]+params
                                    first_list=False
                                else: 
                                    params[0]=input_features
                                out_size=params[1]
                                if initial:
                                    input_features=input_size
                                    initial=False                                
                                self.model.append(nn.Linear(*params))
                                self.model.append(nn.ReLU())
                                in_features=out_size
                            else:
                                print('Parameters neither passed as a dict or as an array')
                            if initial==True:
                                initial=False
                        else:
                            print('Algorithm trying to be added is:',algo)
                            print('The layer cannot be added to the model')
