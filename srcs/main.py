#Import the necessary packages
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle 
# import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import copy
from data_proc import cust_dataset, gener_ds, stft_normalize
from train import train_model

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_names = ["resnet50", "resnet101", "densenet161"] #Deleted "alexnet", "vgg16", "vgg19", "squeezenet"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32 #set 32

# Number of epochs to train for (30<e<50)
num_epochs = 40  #set 40

# Flag for feature extracting. When False, we finetune the whole model,
#when True we only update the reshaped layer params
feature_extract = False

#Normalization values
# mx = 0.0005127093647081057
# mn = 0
# slope = 2/(mx-mn)

#File with the dataset
file_name = 'PhysionetRR.pickle'


num_workers = 0

#Function for initializing models
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet101":
        """ resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet161":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # elif model_name == "inception":
        # """ Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        # """
        # model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs,num_classes)
        # input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Print the model we just instantiated
# print(model_ft)

#Set the loss functino
criterion = nn.CrossEntropyLoss()

#List of subjects
val_ss = [1,3,7] #Random set of subjects

for model_name in model_names:
    for ss in val_ss:
        # torch.cuda.empty_cache()

        #Initialize datasets and dataloaders
        dataasets_dict, dataloaders_dict = gener_ds(file_name, batch_size, num_workers, ss)
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
        #Send model to device GPU
        model_ft = model_ft.to(device)
    
        #Create an optimizer
        params_to_update = model_ft.parameters()
        optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
    
        #Start training
        tr_model, val_acc_hist, train_acc_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs, is_inception=False)
    
        #Create a file to save the model
        # tr_model = tr_model.to(torch.device("cpu"))
        # pickle_out_m = open(model_name + "_model_valsubject_" + ss +  ".pickle","wb")
        # pickle.dump(tr_model, pickle_out_m)
        # pickle_out_m.close()

        #Create a file to save the accuracy history
        pickle_out_acc = open(model_name + "_accuracy_valsubject_" + str(ss) +  ".pickle","wb")
        #Change the device back to cpu
        for d in range(len(val_acc_hist)):
            val_acc_hist[d] = val_acc_hist[d].to(torch.device("cpu"))
            train_acc_hist[d] = train_acc_hist[d].to(torch.device("cpu"))

        pickle.dump([val_acc_hist, train_acc_hist], pickle_out_acc)
        pickle_out_acc.close()
        
    
