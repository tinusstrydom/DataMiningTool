# Tinus Strydom, 2019/08/20
# Data Mining program removing all useful and unknown information

#imports modules/packages
import pandas as pd
import numpy as np

#functions
def dataSet():
    dataSet = pd.read_csv("winequality-red.csv")
    data, target = dataSet[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']], dataSet[['quality']]
    return data, target



#main function
def main():
    data, target = dataSet()
    print(data, target)

if __name__ == "__main__" :
    main()
