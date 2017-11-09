import datetime,nltk,os, numpy as np
import logging,random
from data_conf import PROJECT_FOLDER, event_type_index, realis_index

import multiprocessing

from sequence_detection import get_dataset
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

def mp_worker(dataset):
    clf = KNeighborsClassifier(4).fit(dataset[0],dataset[1])
    return clf

def mp_handler(data):
    p = multiprocessing.Pool(3)
    clfs = p.map(mp_worker, data)
    return clfs

def metacost(clf,m=10,n=100,r=2,
             training_tbf_filename="data/LDC2016E130_training.tbf",
             test_tbf_filename="data/LDC2016E130_test.tbf"):
    X_train,y_train,IDS,_,_ = get_dataset(training_tbf_filename,training=True)
    X_test,y_test,IDS_test,events,corefs = get_dataset(test_tbf_filename,training=False)
    #print(neigh.predict(X[0:10]))    #print(neigh.predict_proba(X[0:10]))    #score = clf.score(X_test, y_test)

    # resample m subset
    data = []
    for i in range(m):
        X, Y = [],[]
        for j in range(n):
            ind = random.randint(0, len(X_train)-1)
            X.append(np.array(X_train[ind],dtype=float))
            Y.append(y_train[ind])
        data.append([X,Y])

    print("Training ...")
    # iterate over classifiers
    p = multiprocessing.Pool(3)
    clfs = p.map(mp_worker, tuple(data))


    C = np.array([[0,1000*r],
                  [1000,0]])
    y_pred = list(range(len(X_test)))
    P_j_x = list(range(2))
    for ind in range(len(X_test)):
        P_j_x[0] = sum([clfs[i].predict([X_test[ind]])[0] == 0  for i in range(m)]) / m
        P_j_x[1] = sum([clfs[i].predict([X_test[ind]])[0] == 1  for i in range(m)]) / m
        y_pred[ind] = np.argmin([ P_j_x[0]*C[0,0] + P_j_x[1]*C[0,1],
                                  P_j_x[0]*C[1,0] + P_j_x[1]*C[1,1]])
    import ipdb ; ipdb.set_trace()
    post_process_predictions(y_pred,IDS_test,events,corefs,name)
    precision,recall,f1 = precision_score(y_test,y_pred), recall_score(y_test,y_pred), f1_score(y_test,y_pred)
    print("%s: %.4f %.4f %.4f" %(name,precision,recall,f1))
