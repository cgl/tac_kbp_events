import nltk,os,pandas, numpy as np
from data_conf import PROJECT_FOLDER, event_type_index, realis_index

from optparse import OptionParser

SOURCE_FOLDER = os.path.join(PROJECT_FOLDER,"data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/")
training_folder = os.path.join(SOURCE_FOLDER,"training")

"""
python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/training/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_training

python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/test/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test

python sequence_detection.py

python2 ~/work/EvmEval/scorer_v1.8.py -a SEQUENCING -g /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test.tbf -s /Users/cagil/work/tac_kbp_events/events/run1_results.txt
"""

def add_links(line,events_doc, corefs_doc, afters_doc,parents_doc):
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
        print(line)
        return


# brat_conversion	1b386c986f9d06fd0a0dda70c3b8ade9	E194	145,154	sentences	Justice_Sentence	Actual
def get_event_dict():
    ANN_FILE = os.path.join(PROJECT_FOLDER,"data/LDC2016E130_test.tbf")
    events, corefs, afters,parents = {},{},{},{}
    with open(ANN_FILE) as ann_file:
        for line in ann_file:
            if line.startswith("#B"):
                doc_id = line.strip().split(" ")[-1]
                events[doc_id] = {}
                corefs[doc_id] = {}
                afters[doc_id] = {}
                parents[doc_id] = {}
            elif line.startswith("@"):
                add_links(line,events[doc_id], corefs[doc_id], afters[doc_id],parents[doc_id])
            elif line.startswith("b"):
                _ , _, event_id, offsets, nugget, event_type, realis = line.strip().split("\t")
                events[doc_id][event_id] = {"offsets":offsets,
                                            "nugget":nugget,
                                            "event_type":event_type,
                                            "realis":realis}
                #yield doc_id, event_id, offsets, nugget, event_type, realis
            else:
                pass
    return events, corefs, afters,parents

import random
def get_results_random(events, corefs, afters,parents):
    run_id = "run1"
    results_str = []
    for doc_id in events.keys():
        results_str.append("#BeginOfDocument %s" %doc_id)
        for event_id in events[doc_id].keys():
            results_str.append("\t".join([run_id,doc_id,event_id,events[doc_id][event_id]["offsets"],
                                               events[doc_id][event_id]["nugget"],
                                               events[doc_id][event_id]["event_type"],
                                               events[doc_id][event_id]["realis"]]))
        for a in range(1,4):
            try:
                key1 = random.choice(list(events[doc_id].keys()))
                events[doc_id].pop(key1)
                key2 = random.choice(list(events[doc_id].keys()))
                results_str.append("@After\tR11\t%s,%s" % (key1,key2))
            except:
                pass
        results_str.append("#EndOfDocument")
    print("\n".join(results_str),file=open("%s_results.txt" %run_id,"w"))

# 'E211' : {'offsets': '1190,1196', 'nugget': 'merged', 'event_type': 'Business_Merge-Org', 'realis': 'Actual'}
def print_after_links_document(doc_id,events_doc, corefs_doc, afters_doc):
    #print("%s\t%s\t%s\t%s" %(doc_id,len(events_doc),len(corefs_doc),len(afters_doc)))
    print(set(events_doc))
    X = []
    Y=[]
    for r_id in afters_doc.keys():
        x = [len(events_doc),len(corefs_doc),]
        for e_id in afters_doc[r_id]:
            nugget = events_doc.get(e_id).get('nugget')
            etype = events_doc.get(e_id).get('event_type')
            offsets = events_doc.get(e_id).get('offsets').split(",")
            realis = events_doc.get(e_id).get('realis')
            print("[%s]%s(%s)" %(e_id,nugget,etype))
            x.append(nugget)
            x.append(event_type_index[etype])
            x.extend(offsets)
            x.append(realis_index[realis])
        print(x)
        X.append(x)
        Y.append(1)
    return X,Y


def get_features(events, corefs, afters,parents):
    run_id = "run1"
    results_str = []
    training_X = []
    training_Y = []
    for doc_id in events.keys():
        X,Y = print_after_links_document(doc_id,events[doc_id],corefs[doc_id],afters[doc_id])
        training_X.extend(X)
        training_Y.extend(Y)
    return training_X,training_Y
def preprocess_dataset(X,y):
    arr_X = np.array(X)
    nuggets = pandas.get_dummies(arr_X[:,2])
    event_types = pandas.get_dummies(arr_X[:,3])
    import ipdb ; ipdb.set_trace()
    arr_X[:,3] = event_types.values.argmax(1)

def main():
    events, corefs, afters,parents = get_event_dict()
    get_results(events, corefs, afters,parents)
    #import ipdb ; ipdb.set_trace()
    #for line in events:
    #    print(line)
def several_classifiers():
    events, corefs, afters,parents = get_event_dict()
    X_training,y_training = get_features(events, corefs, afters,parents)
    X,y = preprocess_dataset(X_training,y_training)
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    print(neigh.predict_proba([[0.9]]))


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-m','--main',default=False,action="store_true",help='')

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
    else:
        several_classifiers()
