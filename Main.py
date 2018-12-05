import numpy as np
import os
import matplotlib.pyplot as plt
import Preprocessor as pp
import Model as mo

Customer_Id = input('Enter Customer Id: ')

Train_GeniuneData = 'data/training/Geniune/'+str(Customer_Id)
Train_ForgedData = 'data/training/Forgeries/'+str(Customer_Id)

Testing_GeniuneData = 'data/training/Geniune/'+str(Customer_Id)
Testing_ForgedData = 'data/training/Forgeries/'+str(Customer_Id)

Training_X = []
Training_Y = []

Testing_X = []
Testing_Y = []

for FileName in os.listdir(Train_GeniuneData):
    Im = plt.imread(os.path.join(Train_GeniuneData, FileName), 0)
    #Preprocessing the Train data for Geniune Signature
    Data = np.array(pp.preprocessing(Im))
    Result = [[0], [1]]
    Result = np.array(Result)
    Training_X.append(list(Data.ravel()))
    Training_Y.append(list(Result.ravel()))

for FileName in os.listdir(Train_ForgedData):
    Im = plt.imread(os.path.join(Train_ForgedData, FileName), 0)
    #Preprocessing the Train data for Forged Signature
    Data = np.array(pp.preprocessing(Im))
    Result = [[1], [0]]
    Result = np.array(Result)
    Training_X.append(list(Data.ravel()))
    Training_Y.append(list(Result.ravel()))

for FileName in os.listdir(Testing_GeniuneData):
    Im = plt.imread(os.path.join(Testing_GeniuneData, FileName), 0)
    #Preprocessing the Test data for Geniune Signature
    Data = np.array(pp.preprocessing(Im))
    Result = [[0], [1]]
    Result = np.array(Result)
    Testing_X.append(list(Data.ravel()))
    Testing_Y.append(list(Result.ravel()))

for FileName in os.listdir(Testing_ForgedData):
    Im = plt.imread(os.path.join(Testing_ForgedData, FileName), 0)
    #Preprocessing the Test data for Forged Signature
    Data = np.array(pp.preprocessing(Im))
    Result = [[1], [0]]
    Result = np.array(Result)
    Testing_X.append(list(Data.ravel()))
    Testing_Y.append(list(Result.ravel()))

Training_X = np.array(Training_X)
Training_Y = np.array(Training_Y)
Testing_X = np.array(Testing_X)
Testing_Y = np.array(Testing_Y)

mo.LogisticRegression(Training_X, Training_Y, Testing_X, Testing_Y)
