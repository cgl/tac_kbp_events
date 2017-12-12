from collections import defaultdict

import datetime, os, numpy as np
import logging, random
from events.data_conf import PROJECT_FOLDER, realis_index
from optparse import OptionParser

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

from data_loader import get_dataset

# References:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

"""
python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/training/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_training

python2 ~/work/EvmEval/util/brat2tbf.py -d /Users/cagil/work/tac_kbp_events/data/LDC2016E130_DEFT_Event_Sequencing_After_Link_Parent_Child_Annotation_Training_Data_V4/data/test/ -o /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test

python sequence_detection.py

python2 ~/work/EvmEval/scorer_v1.8.py -a SEQUENCING -g /Users/cagil/work/tac_kbp_events/data/LDC2016E130_test.tbf -s /Users/cagil/work/tac_kbp_events/events/run1_results.txt
"""


def write_results_tbf(events, afters, run_id="run1"):
    results_str = []
    for doc_id in events.keys():
        results_str.append("#BeginOfDocument %s" %doc_id)
        for event_id in events[doc_id].keys():
            # put events
            results_str.append("\t".join([run_id, doc_id, event_id,
                                          events[doc_id][event_id]["offsets"],
                                          events[doc_id][event_id]["nugget"],
                                          events[doc_id][event_id]["event_type"],
                                          events[doc_id][event_id]["realis"]]))
            # put after links
        for key1, key2 in afters[doc_id].values():
            results_str.append("@After\tR11\t%s,%s" % (key1, key2))
            # results = write_results_after_links_random(events, corefs, afters,parents)
        results_str.append("#EndOfDocument")
    print("\n".join(results_str), file=open("results/%s_results.txt" %run_id, "w"))


names = ["Nearest Neighbors",
         "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA",
         "Linear SVM", "RBF SVM"]
classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
]


def after_links_as_dictionary(y_pred,IDS_test,events,corefs):
    links_found = [i for i in range(len(y_pred)) if y_pred[i]]
    print("Number of links found %d" %len(links_found))
    afters_pred = defaultdict(dict)
    old_doc_id = ""
    for ind in links_found:
        doc_id = IDS_test[ind][0]
        if old_doc_id != doc_id:
            pairs = defaultdict(set)
        try:
            from_event, to_event = IDS_test[ind][1], IDS_test[ind][2]
            from_event_corefs = corefs[doc_id][events[doc_id][from_event]['coref']]
            to_event_corefs = corefs[doc_id][events[doc_id][to_event]['coref']]
            from_event_coref = from_event_corefs[0] if from_event_corefs else from_event
            to_event_coref = to_event_corefs[0] if to_event_corefs else to_event

            # relation does not exist and reverse relation does not exist
            if to_event_coref not in pairs[from_event_coref] and from_event_coref not in pairs[to_event_coref]:
                pairs[from_event_coref].add(to_event_coref)
                afters_pred[doc_id]["R%d" %ind] = [from_event_coref, to_event_coref] #[IDS_test[ind][1],IDS_test[ind][2]]
        except Exception as e:
            logging.exception(e)
            import ipdb ; ipdb.set_trace()
        old_doc_id = doc_id
    print("Number of links after cyclic cleanup %d" %len(afters_pred))
    return afters_pred


def post_process_predictions(y_pred, IDS_test, events, corefs, name):
    afters_pred = after_links_as_dictionary(y_pred, IDS_test, events, corefs)
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    write_results_tbf(events, afters_pred,
                      run_id="%s-%s" % (name.replace(" ", "-"), timestamp))


def several_classifiers(stats=False):
    X_train, y_train, IDS, _, _ = get_dataset("data/LDC2016E130_training.tbf", stats=stats, training=True)
    X_test, y_test, IDS_test, events, corefs = get_dataset("data/LDC2016E130_test.tbf", stats=stats, training=False)
    # print(neigh.predict(X[0:10]))    #print(neigh.predict_proba(X[0:10]))    #score = clf.score(X_test, y_test)
    print("Training ...")
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        post_process_predictions(y_pred, IDS_test, events, corefs, name)
        precision, recall, f1 = precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
        print("%s: %.4f %.4f %.4f" %(name, precision, recall, f1))


def sequence_cnn(stats=False):
    import pickle
    prep = pickle.load(open("prep", "rb"))
    from keras.models import load_model
    model = load_model('mymodel.h5')
    X_test, y_test, IDS_test, events, corefs = get_dataset("data/LDC2016E130_test.tbf", stats=stats, training=False)
    y_pred_prob = model.predict(prep.x_test)
    y_pred = [np.argmax(pred) for pred in y_pred_prob]
    y_test = [np.argmax(y) for y in prep.y_test]

    post_process_predictions(y_pred, IDS_test, events, corefs, "cnn")
    precision, recall, f1 = precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
    print("%s: %.4f %.4f %.4f" %("cnn", precision, recall, f1))


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option('--metacost', default=False, action="store_true", help='')
    parser.add_option('-c', '--cnn', default=False, action="store_true", help='')
    parser.add_option('-s', '--statistics', default=False, action="store_true", help='')
    parser.add_option('-o', '--statsonly', default=False, action="store_true", help='')
    parser.add_option('-d', '--debug', default=False, action="store_true", help='')
    #parser.add_option('-d', '--debug', default=False, action="store_true", help='')

    parser.add_option("-f", "--file", dest="filename",
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="donß't print status messages to stdout")
    parser.add_option('--import_event', default=None, type=int, metavar='FB_ID')
    (options, args) = parser.parse_args()
    if options.metacost:
        from metacost import metacost
        metacost(classifiers[0])
    elif options.statsonly:
        for filename in ["data/LDC2016E130_training.tbf",
                         "data/LDC2016E130_test.tbf", "data/Sequence_2017_test.tbf"]:
            get_dataset(filename, stats=True, training=False)
    elif options.cnn:
        sequence_cnn()
    else:
        several_classifiers(stats=options.statistics)
