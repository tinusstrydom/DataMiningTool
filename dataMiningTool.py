# Tinus Strydom
# Data Mining program removing all useful and unknown information

#imports modules/packages
import pandas as pd
import numpy as np
from pylab import plot,figure,subplot,hist,xlim,show,xlabel,ylabel,title 

#functions
def overviewdata():
    dataSet = pd.read_csv("wine.csv", delimiter=';')
    data, target = dataSet[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']],dataSet[['quality']]
    qltylist = []
    for i in target.quality:
        if not i in qltylist:
            qltylist.append(i)
            qltylist.sort()
    return data, target, dataSet.info(), qltylist

def plotdata(data, target, qltylist):
    #print(data.fixed_acidity.loc[target.quality == 3])
    xlabel('Free Sulfur Dioxide')
    ylabel('Total Sulfur Dioxide')
    title('Relation of sulfur dioxide on the quality')
    color = ['ro','go','bo','mo','co','ko','yo']
    colorIt = iter(color)
    for i in qltylist:
        plot(data.free_sulfur_dioxide.loc[target.quality == i],data.total_sulfur_dioxide.loc[target.quality == i],next(colorIt))
    show()

def histodata(data, target, qltylist):
    xmin = min(data.alcohol)
    xmax = max(data.alcohol)
    count = 421
    color = ['r','g','b','m','c','k','y']
    colorIt = iter(color)
    
    figure(figsize=(10,10),tight_layout=True)
    
    for i in qltylist:
        subplot(count)
        title('Relationship of alcohol and the quality of wine')
        hist(data.alcohol.loc[target.quality==i],color=next(colorIt),alpha=.7,)
        title('Quality = '+str(i))
        xlabel("Alcohol")
        ylabel("Quality")
        xlim(xmin,xmax)
        count+=1  
    show()
    
#main function
def main():
    print('Welcome to data mining tool!')
    print('Lets see an overview of the data\n')
    #View overview of data
    data, target, dataSet, qltylist = overviewdata()
    #Visualization
    print('\nPlotting relation of sulfur dioxide on the quality')
    plotdata(data, target, qltylist)
    print('\nPlot histogram of alcohol on the quality')
    histodata(data, target, qltylist)

    #Classify and cluster

    
    
    #discover relationships
    
    
    #compress
    
    
    #analyse data
    
    
    
if __name__ == "__main__" :
    main()
