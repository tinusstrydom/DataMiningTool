# Tinus Strydom
# Data Mining program removing all useful and unknown information

#imports modules/packages
import pandas as pd
import numpy as np
from pylab import plot,figure,subplot,hist,xlim,show,xlabel,ylabel,title
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,completeness_score,homogeneity_score
from sklearn import metrics
from sklearn.cluster import KMeans



#functions
def overviewdata():
    dataSet = pd.read_csv("wine.csv", delimiter=';')
    data, target = dataSet[['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']],dataSet[['quality']]
    qltylist = []
    for i in target.quality:
        if not i in qltylist:
            qltylist.append(i)
            qltylist.sort()
    print(qltylist)
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

def classify(data, target, qltylist):
    t = np.zeros(len(target))
    count = 1
    for i in qltylist:
        t[target.quality == i] = count
        count+=1
    
    X_train, X_test, y_train, y_test = train_test_split(data, t, test_size=0.3, random_state=109)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("\nAccuracy of Naive Bayes Classification: ")
    print(metrics.accuracy_score(y_test, y_pred))
    #Confusion Matrix
    print("\nConfusion Matrix: ")
    print(confusion_matrix(gnb.predict(X_test),y_test))
    #Classification report
    print("\nReport of performance of the classifier: ")
    print(classification_report(gnb.predict(X_test),y_test, target_names=['3', '4', '5','6','7','8','9']))
    print("\nFollowing 2 Lines help understanding the ZERO division warning due to last label not having enough samples")
    print(np.unique(y_pred))
    print(np.unique(y_test))
    print("\n Average value of the predictions")
    scores = cross_val_score(gnb, data,t, cv=5)
    print(scores)
    print("Mean value of scores:",np.mean(scores))
    return t
    
def clustering(data,t):
    kmeans = KMeans(n_clusters=7, init='random')
    kmeans.fit(data)
    c = kmeans.predict(data)
    print(completeness_score(t,c))
    print(homogeneity_score(t,c))
    print(c)
    figure()
    subplot(211)
    plot(data.free_sulfur_dioxide.loc[t==1],data.total_sulfur_dioxide.loc[t==1],'ro')
    plot(data.free_sulfur_dioxide.loc[t==2],data.total_sulfur_dioxide.loc[t==2],'go')
    plot(data.free_sulfur_dioxide.loc[t==3],data.total_sulfur_dioxide.loc[t==3],'bo')
    plot(data.free_sulfur_dioxide.loc[t==4],data.total_sulfur_dioxide.loc[t==4],'mo')
    plot(data.free_sulfur_dioxide.loc[t==5],data.total_sulfur_dioxide.loc[t==5],'co')
    plot(data.free_sulfur_dioxide.loc[t==6],data.total_sulfur_dioxide.loc[t==6],'ko')
    plot(data.free_sulfur_dioxide.loc[t==7],data.total_sulfur_dioxide.loc[t==7],'yo')
    subplot(212)
    plot(data.free_sulfur_dioxide.loc[c==1],data.total_sulfur_dioxide.loc[c==1],'ro')
    plot(data.free_sulfur_dioxide.loc[c==2],data.total_sulfur_dioxide.loc[c==2],'go')
    plot(data.free_sulfur_dioxide.loc[c==3],data.total_sulfur_dioxide.loc[c==3],'bo')
    plot(data.free_sulfur_dioxide.loc[c==4],data.total_sulfur_dioxide.loc[c==4],'mo')
    plot(data.free_sulfur_dioxide.loc[c==5],data.total_sulfur_dioxide.loc[c==5],'co')
    plot(data.free_sulfur_dioxide.loc[c==6],data.total_sulfur_dioxide.loc[c==6],'ko')
    plot(data.free_sulfur_dioxide.loc[c==7],data.total_sulfur_dioxide.loc[c==7],'yo')
    show()
    
    

    
#main function
def main():
    print('Welcome to data mining tool!')
    
    #View overview of data
    print('Lets see an overview of the data\n')
    data, target, dataSet, qltylist = overviewdata()
    
    #Visualization
    print('\nPlotting relation of sulfur dioxide in comparison to quality')
    #plotdata(data, target, qltylist)
    print('\nPlot histogram of amounts alcohol in comparison to quality')
    #histodata(data, target, qltylist)

    #Classify
    print('\nLets train a classifier from the quality of the wines')
    t = classify(data, target, qltylist)
    
    #Clustering
    print("\nKmeans")
    clustering(data, t)
    
    
    #discover relationships
    
    
    #compress
    
    
    #analyse data
    
    
    
if __name__ == "__main__" :
    main()
