# Tinus Strydom
# Data Mining program removing all useful and unknown information

#imports modules/packages
import pandas as pd
import numpy as np
import pylab as pl



#functions
def overviewdata():
    dataSet = pd.read_csv("wine.csv", delimiter=';')
    data, target = dataSet[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']],dataSet[['quality']]
    return data, target, dataSet.info()

 
def plotdata(data, target):
    #print(data.fixed_acidity.loc[target.quality == 3])
    pl.xlabel('Free Sulfur Dioxide')
    pl.ylabel('Total Sulfur Dioxide')
    pl.title('Relation of sulfur dioxide on the quality')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 3],data.total_sulfur_dioxide.loc[target.quality == 3],'ro')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 4],data.total_sulfur_dioxide.loc[target.quality == 4],'go')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 5],data.total_sulfur_dioxide.loc[target.quality == 5],'bo')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 6],data.total_sulfur_dioxide.loc[target.quality == 6],'mo')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 7],data.total_sulfur_dioxide.loc[target.quality == 7],'co')
    pl.plot(data.free_sulfur_dioxide.loc[target.quality == 8],data.total_sulfur_dioxide.loc[target.quality == 8],'ko')
    pl.show()

def histodata(data, target):
    #use the alcohol column to see 
    xmin = min(data.alcohol)
    xmax = max(data.alcohol)
    pl.figure()
    pl.subplot(611)
    pl.hist(data.alcohol.loc[target.quality==3],color='r',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.subplot(612)
    pl.hist(data.alcohol.loc[target.quality==4],color='g',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.subplot(613)
    pl.hist(data.alcohol.loc[target.quality==5],color='b',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.subplot(614)
    pl.hist(data.alcohol.loc[target.quality==6],color='m',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.subplot(615)
    pl.hist(data.alcohol.loc[target.quality==7],color='c',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.subplot(616)
    pl.hist(data.alcohol.loc[target.quality==8],color='k',alpha=.7)
    pl.xlim(xmin,xmax)
    pl.show()

#main function
def main():
    print('Welcome to data mining tool!')
    print('Lets see an overview of the data\n')
    #View overview of data
    data, target, dataSet = overviewdata()

    #Visualization
    print('\nLets plot and visualize the data')
    plotdata(data, target)
    histodata(data, target)

    #Classify and cluster

    
    
    #discover relationships
    
    
    #compress
    
    
    #analyse data
    
    
    
if __name__ == "__main__" :
    main()
