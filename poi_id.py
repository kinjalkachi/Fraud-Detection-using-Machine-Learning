#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

#import enron 
#import evaluate
import numpy
numpy.random.seed(55)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## Create features list

# features_list = ['poi','salary'] # You will need to use more features

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 
                 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'from_poi_to_this_person', 
                 'exercised_stock_options', 'from_messages', 'other',
                 'from_this_person_to_poi', 
                 'long_term_incentive', 'shared_receipt_with_poi', 
                 'restricted_stock','director_fees'] 
financial_features = ['poi', 'salary', 'total_payments', 'bonus',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive'
                      , 'restricted_stock']
#'email_address',
email_features = ['poi', 'to_messages', 'from_poi_to_this_person',
                  'from_messages', 
                  'from_this_person_to_poi', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Load POI names file
file_poi = open("poi_names.txt", "r")

#****************Data Exploration******************************#

#Total number of data points
#People in the dataset
enron_ppl = len(data_dict)
print "\nnumber of people in enron\n",enron_ppl

#  Number of features used


feature_names = []

print "\nFinancial features:\t\n", len(financial_features)
print "Email features:\t", len(email_features)
for person, features in data_dict.iteritems():
    print"Number of features used are :"
    print(len(features))
    for i in features:
       feature_names.append(i) 
    break
 
print"\nNames of the features used:\n"
print(feature_names)   

#Allocation across classes (POI/non-POI) and  Finding missing values  
# Count of poi and non poi.

# Creating a missing value map:
# In case of missing values it can happen that the values are not entered by mistake or
# people might not have opted or are not eligible for it ,
# In case of "bonus ", some candidate would not be eligible for it 
#or like "deferral_payments" candidate would not have opted for it.
missing_value_map = { 'bonus': {'count':0 }, 'deferral_payments': {'count':0},
    'deferred_income': {'count':0},'director_fees': {'count':0},
    'exercised_stock_options': {'count':0}, 'total_payments': {'count':0},
    'expenses': {'count':0}, 'loan_advances': {'count':0},
    'long_term_incentive': {'count':0}, 
    'restricted_stock_deferred': {'count':0},
    'other': {'count':0, 'poi':0}, 'restricted_stock': {'count':0, 'poi':0},
    'total_stock_value': {'count':0}, 'salary': {'count':0},
    'email_address': {'count':0}, 'from_messages': {'count':0},
    'from_poi_to_this_person': {'count':0},
    'shared_receipt_with_poi': {'count':0},
    'from_this_person_to_poi': {'count':0}, 'to_messages': {'count':0} }

poi_count = 0
nonpoi_count = 0
name_poi = []
totalpay_poi = []



for person, features in data_dict.iteritems():
    if features['poi'] == True:
        poi_count += 1
        name_poi.append(person)
        totalpay_poi.append(features['total_payments'])
    else:
        nonpoi_count += 1
        for name, value in features.iteritems():
            if value == 'NaN':
                missing_value_map[name]['count'] += 1

       
#Find features with more than 50% of missing values
missing_ratio = float
significant_missing_values = []
for feature, values in missing_value_map.iteritems():
    missing_ratio = float(values['count'])/(enron_ppl)
    if missing_ratio > 0.5:
        significant_missing_values.append(feature)
print "\nFeatures with >50% missing values:\n", significant_missing_values


print "\nPerson of interset are:",poi_count
print "\nName of Person of interset are:",name_poi
print "\nOther than Person of interset are:",nonpoi_count

#When I was going through the enron61702insiderpay.pdf 
#I observed that Pois enjoyed large amount of total payment so let us have a look at that too.
print"\n Total payments are in USD."

totalpay_names = {}
for i in range(len(name_poi)):
    totalpay_names[name_poi[i]] = totalpay_poi[i]

print(totalpay_names)


  
### Task 2: Remove outliers
###response identifies outlier(s) in the financial data, 
##and explains how they are removed or otherwise handled
### Task 3: Create new feature(s)



###TOTAL column as the major outlier in this dataset,
# Looking the the XLS we found it is the EXCEL artifact and should be removed. 
###Another outlier is also determined, 
#THE TRAVEL AGENCY IN THE PARK this record did not represent an individual. Both of these should be removed.


## I have choosen exercised stock options and bonus as these are two features that can be exercised by pois.
features2 = ['exercised_stock_options' ,'bonus']
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


data2 = featureFormat( data_dict, features2, sort_keys = True)



from pprint import pprint

exercised_stcoptns_cleaned = []
bonus_cleaned = []
for key in data_dict:
    val = data_dict[key]['exercised_stock_options']
    val1 = data_dict[key]['bonus']

    if (val == 'NaN') :
        continue
    exercised_stcoptns_cleaned.append((key,int(val)))
    if (val1 == 'NaN'):
        continue
    bonus_cleaned.append((key,int(val1)))
    
print "\nPeople who had large exercised stock price"  
pprint(sorted(exercised_stcoptns_cleaned,key=lambda x:x[1],reverse=True)[:2])
print "\nPeople who had large bonus"  
pprint(sorted(bonus_cleaned,key=lambda x:x[1],reverse=True)[:2])

# Even though LAY KENNETH L, HIRKO JOSEPH, LAVORATO JOHN J are outliers , they should be kept because,
# they give a useful piece of information


import matplotlib.pyplot

for point in data2:
    exercised_stcoptns = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( exercised_stcoptns , bonus )

matplotlib.pyplot.xlabel("exercised_stcoptns")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


###since 'from_poi_to_this_person' and 'from_this_person_to_poi' are very important features too, we shoudl investigate those w.r.t. outliers and remove any outliers.

features1 = ["from_this_person_to_poi", "from_poi_to_this_person"]
data1 = featureFormat(data_dict, features1)

### your code below
for point in data1:
    from_this_person_to_poi = point[0]
    from_poi_to_this_person = point[1]
    matplotlib.pyplot.scatter( from_this_person_to_poi, from_poi_to_this_person )

matplotlib.pyplot.xlabel("from_this_person_to_poi")
matplotlib.pyplot.ylabel("from_poi_to_this_person")
matplotlib.pyplot.show()

# finding outliers
to_poi_outliers = []
from_poi_outliers = []
for key in data_dict:
    val = data_dict[key]['from_this_person_to_poi']
    val1 = data_dict[key]['from_poi_to_this_person']
    if val == 'NaN':
        continue
    to_poi_outliers.append((key,int(val)))
    if (val1 == 'NaN'):
        continue
    from_poi_outliers.append((key,int(val1)))
print "\n Outliers in 'to_poi_outliers' are :"
pprint(sorted(to_poi_outliers,key=lambda x:x[1],reverse=True)[:2])
print "\n Outliers in 'from_poi_outliers' are :"
pprint(sorted(from_poi_outliers,key=lambda x:x[1],reverse=True)[:2])


### Updated work : As suggested by reviewer I have used 
# RFECV library for effective selection number and names of features 

# The classes in the sklearn.feature_selection module can be used for feature selection/dimensionality 
#reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance 
#on very high-dimensional datasets.

  
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
lr = LogisticRegression()

rfecv = RFECV(estimator=lr, step=1, cv=StratifiedKFold(labels, 50),
          scoring='precision')
rfecv.fit(features, labels)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Payment ratio of poi = salary/total_payment, gives the ratio of above values.
#popping out outliers:
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


# Engineered features , this could give some more insights on usage of available data:
   
# Extra_payment of poi = total_payment -salary , this shows how much more any poi have than his salary .

for keys,features in data_dict.items():
   tp = features['total_payments'] 
   sal = features['salary']
   if tp == 'NaN' or sal == 'NaN':
       features['extra_pay_poi'] ='NaN'
   else :
        features['extra_pay_poi'] = float(tp)-float(sal)
        
# adding new feature .
features_list +=['extra_pay_poi']

print "\ntotal number of features in including new features:",\
len(features_list)


##Use scikit-learn's SelectKBest feature selection and adding it into my_feature_list.

import operator
from sklearn.feature_selection import SelectKBest,f_classif
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
n=15
new_list = []
k_best = SelectKBest(f_classif,k=n)
k_best.fit_transform(features, labels)
for i in k_best.get_support(indices=True):
     new_list.append(features_list[i+1])

features_score = zip(features_list[1:25],k_best.scores_[:24])
features_score = sorted(features_score,key=operator.itemgetter(1),reverse=True)


print "\nScores of the features :\n"
for i in features_score:
    print i


print "\n New feature list with best 15 features ,  and 1 newly created feature"
print (new_list)
print "\ntotal number of features in including new features:"
print(len(new_list))



#UPDATED WORK : PERFORMING FEATURE IMPORTANCE ON LIST OF FEATURES "new_list":
print "\n Performance with new feature extra_pay_poi"   

new_data = featureFormat(data_dict, new_list)
new_labels, new_features = targetFeatureSplit(new_data)

from sklearn import cross_validation
new_features_train, new_features_test, new_labels_train, new_labels_test = cross_validation.train_test_split(new_features, new_labels, test_size=0.1, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_features_train,new_labels_train)
pred = clf.predict(new_features_test)
from sklearn.metrics import f1_score
f1 = f1_score(pred,new_labels_test , average ='micro' )
print "\nF1 score using 15 best features including newly created feature is \n"
print(f1)


importances = clf.feature_importances_ # this is an iterable object

imp_threshold = 0
ind_imp = []
ls = importances.tolist()

print "\n All importances"
for i in ls:
    print i
    
for i in ls: # this starts the iteration through importances
    if i > 0.2: # if the importance, i, is > 0.2 print it
       imp_threshold = i # appending feature greater than 0.2
       ind_imp.append(ls.index(imp_threshold))
 
# 0.2--Is the threshold that I have set .
#if all words were equally important, each one would give an importance of far less than 0.01    

print "\n Printing most importanct feature with threshold <0.2\n"
for i in ind_imp:
    if i >0.2:
        print  new_list[i] , ":" , ls[i]
    else :
        print "No features with importance > 0.2"
'''
In general, features with higher importance scores are more sensitive to random shuffling 
of their values, which means they are more ‘important’ for prediction. 
Beware that the shuffling is performed for one feature at a time, and although
a feature might seem unnecessary or less important because of its low (or negative) importance score, it
could be the case that it is correlated to other features that can still produce a ‘good’ performance result.
'''
    
print "\n Performance without new feature extra_pay_poi:"   
new_list.remove('extra_pay_poi')
print "\n extra_pay_poi popped out using .remove"

new_data = featureFormat(data_dict, new_list)
new_labels, new_features = targetFeatureSplit(new_data)

from sklearn import cross_validation
new_features_train, new_features_test, new_labels_train, new_labels_test = cross_validation.train_test_split(new_features, new_labels, test_size=0.1, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_features_train,new_labels_train)
pred = clf.predict(new_features_test)
from sklearn.metrics import f1_score
f1 = f1_score(pred,new_labels_test , average ='micro' )
print "\nF1 score using 15 best features without newly created feature is \n"
print(f1)




 


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

###https://www.quora.com/How-do-I-properly-use-SelectKBest-GridSearchCV-and-cross-validation
###-in-the-sklearn-package-together


# Classification. When the data are being used to predict a category, 
#supervised learning is also called classification.
print "\n FINAL FEATURES LIST USED:\n"
print(features_list)
my_data = featureFormat(data_dict, features_list)
gs_labels, gs_features = targetFeatureSplit(my_data)

from sklearn import cross_validation

my_features_train, my_features_test, my_labels_train, my_labels_test = cross_validation.train_test_split(gs_features, gs_labels, random_state=42,test_size=0.1)

from tester import test_classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


print "\nGaussianNB classifier(Default) :"
NB = GaussianNB()
test_classifier(NB,data_dict,features_list,folds = 1000)

print "\nDecission Tree classifier :"
dec_tree = DecisionTreeClassifier()
test_classifier(dec_tree,data_dict,features_list,folds = 1000)

print "\n  KNeighbors Classifier :"
k_neighb  =  KNeighborsClassifier(5)
test_classifier(k_neighb ,data_dict,features_list,folds = 1000)

# As you can see KNeighborsClassifier shows maximum accuracy and precision ,but high accuracy is not always good..

print "\n param_grid with k_neighbours"


import numpy as np
k = np.arange(7)+1
parameters = {'n_neighbors': k}
import sklearn
knn = sklearn.neighbors.KNeighborsClassifier()

#The grid search provided by GridSearchCV exhaustively generates candidates from 
#a grid of parameter values specified with the param_grid parameter.

grid_search = GridSearchCV(knn, param_grid = parameters)
grid_search.fit( my_features_train,  my_labels_train)
clf = grid_search.best_estimator_

test_classifier(clf,data_dict,features_list,folds = 1000)


# As you can see using we get more accuracy and recall using GridSearchCV.
# We see that precision is higher than recall ,
# High precision, low recall

#Hence ,if we have an algorithm with high precision, we can trust the classification judgements made by it. 
#In this example, the algorithm ran has maximum precision, since the one piece of content that it did label was in fact spammy.



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

my_data = featureFormat(data_dict, features_list)
gs_labels, gs_features = targetFeatureSplit(my_data)

from sklearn import cross_validation 

my_features_train, my_features_test, my_labels_train, my_labels_test = cross_validation.train_test_split(gs_features, gs_labels, random_state=42,test_size=0.1)
shuffle= cross_validation.StratifiedShuffleSplit( my_labels_train,n_iter = 25,test_size = 0.5,
                                random_state = 0)
print "\n PCA with GaussianNb \n"
param_grid = {
         'pca__n_components':[1,2,3,4,5,6,7,8,9,10]
          }
estimators = [('pca',PCA()),('gaussian',GaussianNB())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1', cv = shuffle)
gs.fit(my_features_train,  my_labels_train)

clf = gs.best_estimator_
test_classifier(clf,data_dict,features_list,folds = 1000)


print "\n PCA with decisiontree \n"
param_grid = {
         'pca__n_components':[1,2,3,4,5,6],
         'tree__min_samples_split':[2,5,10,100],
         'tree__criterion':['gini'],
         'tree__splitter':['best']
          }
estimators = [('pca',PCA()),('tree',DecisionTreeClassifier())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1',cv=shuffle)
gs.fit(my_features_train,  my_labels_train)
clf = gs.best_estimator_
test_classifier(clf,data_dict,features_list,folds = 1000)







### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, features_list)
