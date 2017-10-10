from collections import defaultdict

import datetime,nltk,os, numpy as np
from data_conf import PROJECT_FOLDER, event_type_index, realis_index

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import recall_score, precision_score, f1_score

from optparse import OptionParser

import random

"""
python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/training/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_training

python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/test/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test

python sequence_detection.py

python2 ~/work/EvmEval/scorer_v1.8.py -a SEQUENCING -g /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test.tbf -s /Users/cagil/work/tac_kbp_events/events/run1_results.txt
"""

def read_relations(line,events_doc, corefs_doc, afters_doc,parents_doc):
    _, lid, event_ids = line.strip().split("\t")
    if line.startswith("@After"):
        afters_doc[lid] = event_ids.split(",")
    elif line.startswith("@Coreference"):
        corefs_doc[lid] = event_ids.split(",")
        for e_id in corefs_doc[lid]:
            events_doc[e_id]["coref"]=lid
    elif line.startswith("@Subevent"):
        parents_doc[lid] = event_ids.split(",")
    else:
        #print(line)
        return

def add_corefs_to_single_events(events,corefs):
    for doc_id in events.keys():
        index = len(corefs[doc_id])
        for event_id in events[doc_id].keys():
            if "coref" not in events[doc_id][event_id]:
                events[doc_id][event_id]["coref"] = "C%d" %index
                corefs[doc_id]["C%d" %index] = [event_id]
                index +=1

# brat_conversion	1b386c986f9d06fd0a0dda70c3b8ade9	E194	145,154	sentences	Justice_Sentence	Actual
def read_annotations(ann_file_tbf):
    events, corefs, afters,parents = {},{},{},{}
    with open(ann_file_tbf) as ann_file:
        for line in ann_file:
            if line.startswith("#B"):
                doc_id = line.strip().split(" ")[-1]
                events[doc_id] = {}
                corefs[doc_id] = {}
                afters[doc_id] = {}
                parents[doc_id] = {}
            elif line.startswith("@"):
                read_relations(line,events[doc_id], corefs[doc_id], afters[doc_id],parents[doc_id])
            elif line.startswith("b"):
                _ , _, event_id, offsets, nugget, event_type, realis = line.strip().split("\t")
                events[doc_id][event_id] = {"offsets":offsets,
                                            "nugget":nugget,
                                            "event_type":event_type,
                                            "realis":realis}
                #yield doc_id, event_id, offsets, nugget, event_type, realis
            else:
                pass
    add_corefs_to_single_events(events,corefs)
    return events, corefs, afters,parents

def write_results_tbf(events,afters,run_id = "run1"):
    results_str = []
    for doc_id in events.keys():
        results_str.append("#BeginOfDocument %s" %doc_id)
        for event_id in events[doc_id].keys():
            # put events
            results_str.append("\t".join([run_id,doc_id,event_id,events[doc_id][event_id]["offsets"],
                                               events[doc_id][event_id]["nugget"],
                                               events[doc_id][event_id]["event_type"],
                                               events[doc_id][event_id]["realis"]]))
            # put after links
        for key1,key2 in afters[doc_id].values():
            results_str.append("@After\tR11\t%s,%s" % (key1,key2))
            #results = write_results_after_links_random(events, corefs, afters,parents)
        results_str.append("#EndOfDocument")
    print("\n".join(results_str),file=open("results/%s_results.txt" %run_id,"w"))

def write_results_after_links_random(events, corefs, afters,parents):
        for a in range(1,4):
            try:
                key1 = random.choice(list(events[doc_id].keys()))
                events[doc_id].pop(key1)
                key2 = random.choice(list(events[doc_id].keys()))
                results_str.append("@After\tR11\t%s,%s" % (key1,key2))
            except:
                pass

def build_feature_vector(linked_events,events_doc,corefs_doc):
    x = [len(events_doc),len(corefs_doc),]
    for e_id in linked_events:
        nugget = events_doc.get(e_id).get('nugget')
        etype = events_doc.get(e_id).get('event_type')
        offsets = events_doc.get(e_id).get('offsets').split(",")
        realis = events_doc.get(e_id).get('realis')
        #print("[%s]%s(%s)" %(e_id,nugget,etype))
        x.append(nugget)
        x.append(event_type_index[etype])
        x.extend([int(x) for x in offsets])
        x.append(realis_index[realis])
    #add offset distance
    [e1_id,e2_id] = linked_events
    x.append(abs(int(events_doc.get(e1_id).get('offsets').split(",")[0])-int(events_doc.get(e2_id).get('offsets').split(",")[0])))
    return x

# 'E211' : {'offsets': '1190,1196', 'nugget': 'merged', 'event_type': 'Business_Merge-Org', 'realis': 'Actual'}
def build_feature_matrix_for_document(doc_id,events_doc, corefs_doc, afters_doc,training=True):
    #print("%s\t%s\t%s\t%s" %(doc_id,len(events_doc),len(corefs_doc),len(afters_doc)))
    #print(set(events_doc))
    X = []
    Y=[]
    IDS = []
    event_id_list = set(events_doc.keys())

    for event_id in event_id_list:
        for to_event_id in event_id_list:
            #import ipdb ; ipdb.set_trace()
            # no link definitions between corefs
            if 'coref' in events_doc[event_id] and to_event_id in corefs_doc[events_doc[event_id]['coref']]:
                continue
            linked_event_ids = [event_id,to_event_id]
            x = build_feature_vector(linked_event_ids,events_doc,corefs_doc)
            is_positive = linked_event_ids in afters_doc.values()
            if training and is_positive:
                pass
            elif x[-1] > 650:
                continue
            X.append(x)
            if is_positive:
                Y.append(1)
            else:
                Y.append(0)
            IDS.append([doc_id,linked_event_ids[0],linked_event_ids[1]])
    return X,Y,IDS

def build_feature_matrix_for_document_old(doc_id,events_doc, corefs_doc, afters_doc,add_neg=True):
    #print("%s\t%s\t%s\t%s" %(doc_id,len(events_doc),len(corefs_doc),len(afters_doc)))
    #print(set(events_doc))
    X,Y,IDS = [],[],[]
    for linked_event_ids in afters_doc.values(): #r_id in afters_doc.keys():
        x = build_feature_vector(linked_event_ids,events_doc,corefs_doc)
        X.append(x)
        Y.append(1)
        IDS.append([doc_id,linked_event_ids[0],linked_event_ids[1]])
    if add_neg:
        event_id_list = events_doc.keys()
        number_of_positive_links = len(X)
        number_of_negative_links = 0
        # add same amount of negative links
        while number_of_negative_links < 4*number_of_positive_links:
            random_ids = random.sample(event_id_list,2)
            if random_ids in afters_doc.values():
                continue
            x = build_feature_vector(random_ids,events_doc,corefs_doc)
            if x[-1] > 650:
                continue
            X.append(x)
            Y.append(0)
            IDS.append([doc_id,random_ids[0],random_ids[1]])
            number_of_negative_links += 1
    if len(afters_doc) != sum(Y):
        print("Afters are missing in document %s (%d vs %d)" %(doc_id,len(afters_doc),sum(Y)))
    return X,Y,IDS

def build_feature_matrix_for_dataset(events, corefs, afters,parents,training=True):
    run_id = "run1"
    results_str = []
    training_X = []
    training_Y = []
    training_IDS = []
    for doc_id in events.keys():
        if training: #old
            X,Y, IDS = build_feature_matrix_for_document(doc_id,events[doc_id],corefs[doc_id],afters[doc_id],training=training)
        else:
            X,Y, IDS = build_feature_matrix_for_document(doc_id,events[doc_id],corefs[doc_id],afters[doc_id],training=training)
        training_X.extend(X)
        training_Y.extend(Y)
        training_IDS.extend(IDS)
    print("%s set: %s possible links between pairs" %("Training" if training else "Test", len(training_X)))
    return training_X,training_Y, training_IDS

def cosine_sim(word_emb1,word_emb2):
    numerator = np.dot(word_emb1,word_emb2)
    denominator = np.sqrt(np.sum(word_emb1**2)) * np.sqrt(np.sum(word_emb2**2))
    #print(numerator,denominator,float(numerator/denominator))
    if numerator and denominator:
        return float(numerator/denominator)
    else:
        return 0

def preprocess_dataset(X):
    arr_X = np.array(X,dtype=object)
    from prepare_datafile import EmbeddingBank
    emb = EmbeddingBank()
    #emb_sim_column = [emb.get_embedding(arr_X[ind,2])-emb.get_embedding(arr_X[ind,7]) for ind in range(arr_X.shape[0])]

    emb_sim_column = [cosine_sim(emb.get_embedding(arr_X[ind,2]),emb.get_embedding(arr_X[ind,7])) for ind in range(arr_X.shape[0])]

    for i in [2,7]:
        emb_column = [emb.get_embedding(arr_X[ind,i]) for ind in range(arr_X.shape[0])]
        ind_column = [emb.get_index(arr_X[ind,i]) for ind in range(arr_X.shape[0])]
        arr_X[:,i] = ind_column
        arr_X = np.append(arr_X,np.array(emb_column),1)

    #arr_X = np.append(arr_X,np.array(emb_sim_column),1)
    arr_X = np.append(arr_X,np.array(emb_sim_column).reshape(len(emb_sim_column),1),1)
    return arr_X

def main(debug=False):
    ann_file_tbf = os.path.join(PROJECT_FOLDER,"data/LDC2016E130_test.tbf")
    if debug:
        import ipdb ; ipdb.set_trace()
    events, corefs, afters,parents = read_annotations(ann_file_tbf)
    get_results(events, corefs, afters,parents)
    #for line in events:
    #    print(line)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

def get_stats(events, corefs, afters, parents,X_train,y_train,IDS):
    number_of_events = sum([len(events[item]) for item in events])
    number_of_docs = len(events)

    number_of_afters = sum([len(afters[item]) for item in afters])

    number_of_unique_events = 0
    for doc_id in events:
        number_of_unique_events += len(corefs[doc_id])
        for event_id in events[doc_id]:
            if "coref" not in events[doc_id][event_id]:
                number_of_unique_events +=1

    print("There are %d number of events in %d documents" %(number_of_events,number_of_docs))
    print("There are %d number of unique events" %(number_of_unique_events))
    print("There are %d number of after links" %(number_of_afters))

# filename = "data/LDC2016E130_training.tbf"
def get_dataset(filename,training=True,stats=False):
    ann_file_tbf = os.path.join(PROJECT_FOLDER,filename)
    events, corefs, afters,parents = read_annotations(ann_file_tbf)
    X_train,y_train,IDS = build_feature_matrix_for_dataset(events, corefs, afters,parents,training=training)
    if stats:
        get_stats(events, corefs, afters, parents,X_train,y_train,IDS )
    X_train = preprocess_dataset(X_train)
    return X_train,y_train,IDS, events,corefs

def after_links_as_dictionary(y_pred,IDS_test,events,corefs):
    links_found = [i for i in range(len(y_pred)) if y_pred[i]]
    afters_pred = defaultdict(dict)

    for ind in links_found:
        doc_id = IDS_test[ind][0]
        from_event, to_event = IDS_test[ind][1],IDS_test[ind][2]
        import ipdb ; ipdb.set_trace()
        events[doc_id]
        afters_pred[doc_id]["R%d" %ind] = [IDS_test[ind][1],IDS_test[ind][2]]
    return afters_pred

def post_process_predictions(y_pred,IDS_test,events,corefs):
    afters_pred =  after_links_as_dictionary(y_pred,IDS_test,events,corefs)
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    write_results_tbf(events, afters_pred,run_id="%s-%s" %(name.replace(" ","-"),timestamp))

def several_classifiers(stats=False):
    X_train,y_train,IDS,_,_ = get_dataset("data/LDC2016E130_training.tbf",stats=stats,training=True)
    X_test,y_test,IDS_test,events,corefs = get_dataset("data/LDC2016E130_test.tbf",stats=stats,training=False)
    #print(neigh.predict(X[0:10]))    #print(neigh.predict_proba(X[0:10]))    #score = clf.score(X_test, y_test)
    print("Training ...")
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        post_process_predictions(y_pred,IDS_test,events,corefs)
        precision,recall,f1 = precision_score(y_test,y_pred), recall_score(y_test,y_pred), f1_score(y_test,y_pred)
        print("%s: %.4f %.4f %.4f" %(name,precision,recall,f1))

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-m','--main',default=False,action="store_true",help='')
    parser.add_option('-s','--statistics',default=False,action="store_true",help='')
    parser.add_option('-d','--debug',default=False,action="store_true",help='')
    #parser.add_option('-d','--debug',default=False,action="store_true",help='')

    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")
    parser.add_option('--import_event',default=None,type=int,metavar='FB_ID',help='')
    (options, args) = parser.parse_args()
    #import ipdb ; ipdb.set_trace()
    if options.main:
        main()
    elif options.debug:
        main(debug=True)
    else:
        several_classifiers(stats=options.statistics)
