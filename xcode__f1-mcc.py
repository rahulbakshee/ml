###############################################################################
# reduce the size of dataframe

test["avg"] = test["avg"].astype('float16')
test["first"] = test["first"].astype('uint8')

###############################################################################
# kurt and skew
# use log1p,log10,log,sqrt,log_sqrt,sqrt_log,StdScaler,MinMaxScaler to handle them
# the values should be as near as possible to zero

print("###skew###", train.skew(), sep='\n')
print("                            ")
print("###kurt###", train.kurt(),sep='\n')

train["don_log"] = np.log10(train["don"])
train["avg_sqrt"] = np.sqrt(train["avg"])

###############################################################################
# without GridSearchCV 
# f1_score, mcc error, oob score for classification
# param tuning of alg based on their f1score, mcc error, oob score
######## 
# The MCC is in essence a correlation coefficient value between -1 and +1. 
# A coefficient of +1 represents a perfect prediction,0 an average random prediction and -1 an inverse prediction. 
######## 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its 
# best value at 1 and worst score at 0. 
### Testing grid for f1 score random forest
x_train, x_test, y_train, y_test = train_test_split(train[predictors],
                                                    train["Made Donation in March 2007"],
                                                    test_size = 0.33,
                                                    random_state=seed)

#f1 = 0
mcc = 0
start_time = time.time()
for esti in [90,100,110]:
    for depth in [4,5,6]:
        for split in [8,9,10,11]:
            for leaf in [1,2,3]:
                for feat in [2,3]:
                    clf = RandomForestClassifier(random_state=seed,n_jobs=-1,bootstrap=True,oob_score=True, n_estimators=esti,
                                           max_depth=depth, min_samples_split=split, min_samples_leaf=leaf,max_features=feat)
                
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_test)

                    if matthews_corrcoef(y_test, pred) > mcc:
                        mcc = matthews_corrcoef(y_test, pred)
                        print("{} {} {} {} {} oob score{}".format(esti,depth,split,leaf,feat,clf.oob_score_))
                        print("f1score", f1_score(y_test, pred))
                        print("mcc",matthews_corrcoef(y_test, pred))
                        accu = metrics.accuracy_score(y_test, pred)
                        print("accu  {}  ".format(accu))
                        tn, fp, fn, tp  = confusion_matrix(y_test,pred).ravel()
                        print(tn, fp, fn, tp )  
                        print("+++++")
                        gc.collect()
                   
###############################################################################
# using GridSearchCV
# grid search for multiple alg using GridSearchCV and all models in a dict

models = {LogisticRegression(random_state=21,n_jobs=-1):{'C':[0.29],
                                                         'fit_intercept':[True],
                                                         'max_iter':[9]},
          RidgeClassifier(random_state=21):{'alpha':[0.075],
                                            'fit_intercept':[False],
                                            'max_iter':[10,100],
                                            'normalize':[True]},
          SGDClassifier(random_state=21,n_jobs=-1):{'fit_intercept':[True],
                                                    'max_iter':[3000],
                                                    'alpha':[0.04]
                                                   },
          LinearSVC(random_state=21):{'fit_intercept':[False],
                                      'C':[0.08],
                                      'max_iter':[1700] },
          SVC(random_state=21):{'C':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5],
                               'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}
                    #results#   best param {'C': 1, 'kernel': 'linear'}
                                best score 0.875777824387
                                accuracy train_test_split 0.766956521739
         AdaBoostClassifier(random_state=seed):{'learning_rate':[0.001,0.01,0.05,0.1,0.5,1,2,3],
                                                'n_estimators':[50,100,200,300,400,500,700,1000,1500,2000,2500] },
          
          
          BaggingClassifier(random_state=seed,n_jobs=-1,bootstrap=True):{'oob_score':[True,False],
                                                                         'max_samples':[0.7,0.8,0.9,1.0],
                                                                         'max_features':[2,3,4],
                                                                         'n_estimators':[50,100,200,300,400,500,700,1000,1500,2000] },
          
          
          ExtraTreesClassifier(random_state=seed,n_jobs=-1,bootstrap=True):{'oob_score':[True,False],
                                                                            'n_estimators':[50,100,200,300,400,500,700,1000,1500,2000],
                                                                            'max_depth':[1,3,5,7,9],
                                                                            'min_samples_split':[2,4,6,8,10],
                                                                            'min_samples_leaf':[1,3,5,7,9] },
          
          GradientBoostingClassifier(random_state=seed):{'learning_rate':[0.01,0.05,0.1,0.5,1,1.5,2],
                                                         'n_estimators':[50,100,200,300,400,500,700,1000,1500,2000],
                                                         'subsample':[0.7,0.8,0.9,1.0],
                                                         'min_samples_split':[2,4,6,8,10],
                                                         'min_samples_leaf':[1,3,5,7,9],
                                                         'max_depth':[3,4,5,6],
                                                         'max_features':[2,3,4] },
          
          RandomForestClassifier(random_state=seed,n_jobs=-1,bootstrap=True):{'oob_score':[True,False],
                                                                              'n_estimators':[50,100,200,300,400,500,700,1000,1500,2000],
                                                                              'max_depth':[1,3,5,7,9],
                                                                              'min_samples_split':[2,4,6,8,10],
                                                                              'min_samples_leaf':[1,3,5,7,9],
                                                                              'max_features':[2,3,4] } 
         }

for esti, param in models.items():
    start_time = time.time()
    print(str(esti)[:15])
    gridsearch = GridSearchCV(estimator=esti,
                          param_grid=param,
                          scoring='roc_auc', 
                          fit_params=None, 
                          n_jobs=-1, 
                          iid=True, 
                          refit=True, 
                          cv=10, 
                          verbose=0, 
                          pre_dispatch='2*n_jobs', 
                          error_score='raise')


    gridsearch.fit(train[predictors], train['Made Donation in March 2007'])
    
    print("best param", gridsearch.best_params_)
    print("best grid score",gridsearch.best_score_)
    try:
        if gridsearch.best_estimator_.oob_score :
            print("oob score", gridsearch.best_estimator_.oob_score_)
    except:
        print("oob not supported")
        pass
    
    print("gridsearch done, starting with the train_test_split accuracy")
    gc.collect()
    
    accu = 0
    total_accu = 0
    for sid in [21,42,37,0,99]:
        x_train, x_test, y_train, y_test = train_test_split(train[predictors],
                                                            train["Made Donation in March 2007"],
                                                            test_size = 0.33,random_state=sid)
        gridsearch.best_estimator_.fit(x_train, y_train)
        pred = gridsearch.best_estimator_.predict(x_test)
        accu = metrics.accuracy_score(y_test, pred)
        total_accu += accu
        print(" {} accu with seed {}".format(accu,sid))
        gc.collect()
    print("mean train_test_split accuracy ", total_accu/5)
    print("total time taken {} minutes".format((time.time()-start_time)/60))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    gc.collect()
gc.collect()
