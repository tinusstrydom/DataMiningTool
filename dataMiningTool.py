# Tinus Strydom
# Data Mining program removing all useful and unknown information

#imports modules/packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy import linspace, matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,completeness_score,homogeneity_score, mean_squared_error
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression



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
    plt.xlabel('Free Sulfur Dioxide')
    plt.ylabel('Total Sulfur Dioxide')
    plt.title('Relation of sulfur dioxide on the quality')
    color = ['ro','go','bo','mo','co','ko','yo']
    colorIt = iter(color)
    for i in qltylist:
        plt.plot(data.free_sulfur_dioxide.loc[target.quality == i],data.total_sulfur_dioxide.loc[target.quality == i],next(colorIt))
    plt.show()

def histodata(data, target, qltylist):
    xmin = min(data.alcohol)
    xmax = max(data.alcohol)
    count = 421
    color = ['r','g','b','m','c','k','y']
    colorIt = iter(color)
    
    plt.figure(figsize=(10,10),tight_layout=True)
    
    for i in qltylist:
        plt.subplot(count)
        plt.title('Relationship of alcohol and the quality of wine')
        plt.hist(data.alcohol.loc[target.quality==i],color=next(colorIt),alpha=.7,)
        plt.title('Quality = '+str(i))
        plt.xlabel("Alcohol")
        plt.ylabel("Quality")
        plt.xlim(xmin,xmax)
        count+=1  
    plt.show()

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
    plt.figure()
    plt.subplot(211)
    plt.plot(data.free_sulfur_dioxide.loc[t==1],data.total_sulfur_dioxide.loc[t==1],'ro')
    plt.plot(data.free_sulfur_dioxide.loc[t==2],data.total_sulfur_dioxide.loc[t==2],'go')
    plt.plot(data.free_sulfur_dioxide.loc[t==3],data.total_sulfur_dioxide.loc[t==3],'bo')
    plt.plot(data.free_sulfur_dioxide.loc[t==4],data.total_sulfur_dioxide.loc[t==4],'mo')
    plt.plot(data.free_sulfur_dioxide.loc[t==5],data.total_sulfur_dioxide.loc[t==5],'co')
    plt.plot(data.free_sulfur_dioxide.loc[t==6],data.total_sulfur_dioxide.loc[t==6],'ko')
    plt.plot(data.free_sulfur_dioxide.loc[t==7],data.total_sulfur_dioxide.loc[t==7],'yo')
    plt.subplot(212)
    plt.plot(data.free_sulfur_dioxide.loc[c==1],data.total_sulfur_dioxide.loc[c==1],'ro')
    plt.plot(data.free_sulfur_dioxide.loc[c==2],data.total_sulfur_dioxide.loc[c==2],'go')
    plt.plot(data.free_sulfur_dioxide.loc[c==3],data.total_sulfur_dioxide.loc[c==3],'bo')
    plt.plot(data.free_sulfur_dioxide.loc[c==4],data.total_sulfur_dioxide.loc[c==4],'mo')
    plt.plot(data.free_sulfur_dioxide.loc[c==5],data.total_sulfur_dioxide.loc[c==5],'co')
    plt.plot(data.free_sulfur_dioxide.loc[c==6],data.total_sulfur_dioxide.loc[c==6],'ko')
    plt.plot(data.free_sulfur_dioxide.loc[c==7],data.total_sulfur_dioxide.loc[c==7],'yo')
    plt.show()
    
def regress(data):
    x = data.total_sulfur_dioxide[:,np.newaxis]
    y = data.free_sulfur_dioxide
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=109)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    
    m = reg.coef_[0]
    b = reg.intercept_
    print("Slope = ",m,"\nintercept = ",b)
    print("Mean squared error = ",mean_squared_error(y_test,y_pred))
    print()
        
    plt.scatter(X_test,y_test,c='black')
    plt.plot(X_test, y_pred, 'b')
    plt.xlabel("Free Sulfur")
    plt.ylabel("Total Sulfur")
    plt.show()
    
def correlation(data):
    corr = np.corrcoef(data.T)
    print(corr)
    plt.pcolor(corr)
    plt.colorbar()
    plt.xticks(np.arange(0.5, 11.5),['fixed_acid','vola_acid','cit_acid','res_sugar','chlorides','free_sulf','tot_sulf','density','pH','sulphates','alcohol'])
    plt.yticks(np.arange(0.5, 11.5),['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol'])
    plt.show()
    
    
    
#main function
def main():
    print('Welcome to data mining tool!')
    
    #Data Importing
    print('Lets see an overview of the data\n')
    data, target, dataSet, qltylist = overviewdata()
    
    #Visualization
    #print('\nPlotting relation of sulfur dioxide in comparison to quality')
    #plotdata(data, target, qltylist)
    #print('\nPlot histogram of amounts alcohol in comparison to quality')
    #histodata(data, target, qltylist)

    #Classification
    #print('\nLets train a classifier from the quality of the wines')
    #t = classify(data, target, qltylist)
    
    #Clustering
    #print("\nKmeans")
    #clustering(data, t)
    
    #Regression
    #print("\nRegression")
    #regress(data)
    
    #Correlation
    print("\n Correlation")
    correlation(data)
    
    #Dimensionality Reduction
    
    #Networks Mining
    
    
    
if __name__ == "__main__" :
    main()
