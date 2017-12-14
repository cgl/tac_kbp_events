import os
from events.data_conf import PROJECT_FOLDER, event_type_index, realis_index
import numpy as np


def read_relations(line, events_doc, corefs_doc, afters_doc, parents_doc):
    _, lid, event_ids = line.strip().split("\t")
    if line.startswith("@After"):
        afters_doc[lid] = event_ids.split(",")
    elif line.startswith("@Coreference"):
        corefs_doc[lid] = event_ids.split(",")
        for e_id in corefs_doc[lid]:
            events_doc[e_id]["coref"] = lid
    elif line.startswith("@Subevent"):
        parents_doc[lid] = event_ids.split(",")
    else:
        pass


def add_corefs_to_single_events(events, corefs):
    for doc_id in events.keys():
        index = len(corefs[doc_id])
        for event_id in events[doc_id].keys():
            if "coref" not in events[doc_id][event_id]:
                events[doc_id][event_id]["coref"] = "C%d" %index
                corefs[doc_id]["C%d" %index] = [event_id]
                index += 1


# brat_conversion 1b386c986f9d06fd0a0dda70c3b8ade9 E194	145,154	sentences Justice_Sentence Actual
def read_annotations(ann_file_tbf):
    events, corefs, afters, parents = {}, {}, {}, {}
    with open(ann_file_tbf) as ann_file:
        for line in ann_file:
            if line.startswith("#B"):
                doc_id = line.strip().split(" ")[-1]
                events[doc_id] = {}
                corefs[doc_id] = {}
                afters[doc_id] = {}
                parents[doc_id] = {}
            elif line.startswith("@"):
                read_relations(line, events[doc_id], corefs[doc_id], afters[doc_id],parents[doc_id])
            elif line.startswith("b"):
                _ , _, event_id, offsets, nugget, event_type, realis = line.strip().split("\t")
                events[doc_id][event_id] = {"offsets": offsets,
                                            "nugget": nugget,
                                            "event_type": event_type,
                                            "realis": realis}
            else:
                pass
    add_corefs_to_single_events(events, corefs)
    return events, corefs, afters, parents
##############################################################################


def build_feature_vector(linked_events, events_doc, corefs_doc):
    x = [len(events_doc),len(corefs_doc),]
    for e_id in linked_events:
        nugget = events_doc.get(e_id).get('nugget')
        etype = events_doc.get(e_id).get('event_type')
        offsets = events_doc.get(e_id).get('offsets').split(",")
        realis = events_doc.get(e_id).get('realis')
        x.append(nugget)
        x.append(event_type_index[etype])
        x.extend([int(x) for x in offsets])
        x.append(realis_index[realis])
    # add offset distance
    [e1_id, e2_id] = linked_events
    x.append(abs(int(events_doc.get(e1_id).get('offsets').split(",")[0]) -
                 int(events_doc.get(e2_id).get('offsets').split(",")[0])))
    return x


def get_coref_links(linked_event_ids, events_doc, corefs_doc, doc_id):
    [from_event, to_event] = linked_event_ids
    from_event_corefs = corefs_doc[events_doc[from_event]['coref']]
    to_event_corefs = corefs_doc[events_doc[to_event]['coref']]
    try:
        from_event_corefs.remove(from_event)
    except ValueError:
        pass
    coref_links = []
    coref_links_negatives = []
    while from_event_corefs:
        fro = from_event_corefs.pop()
        for to in to_event_corefs:
            coref_links.append([fro, to])
            coref_links_negatives.append([to, fro])
    return coref_links, coref_links_negatives


# 'E211' : {'offsets': '1190,1196', 'nugget': 'merged',
#           'event_type': 'Business_Merge-Org', 'realis': 'Actual'}
def build_feature_matrix_for_document(doc_id, events_doc, corefs_doc,
                                      afters_doc, training=True):
    X = []
    Y = []
    IDS = []
    event_id_list = set(events_doc.keys())
    coref_positives, negatives = [], []
    for event_id in event_id_list:
        for to_event_id in event_id_list:
            linked_event_ids = [event_id, to_event_id]
            is_positive = linked_event_ids in afters_doc.values()
            distance = abs(int(events_doc.get(event_id).get('offsets').split(",")[0]) - int(events_doc.get(to_event_id).get('offsets').split(",")[0]))
            # eliminate all pairs with a distance larger than 500 if not (in afters list and training)
            if not (training and is_positive) and distance > 500:
                continue

            # add all annotations and their positive and negative extensions
            if is_positive:
                coref_links, coref_links_negatives = get_coref_links(linked_event_ids,events_doc, corefs_doc,doc_id)
                coref_positives.extend(coref_links)
                negatives.extend(coref_links_negatives)
                negatives.append(linked_event_ids[::-1])
                x = build_feature_vector(linked_event_ids,events_doc,corefs_doc)
                X.append(x)
                Y.append(1)
                IDS.append([doc_id,linked_event_ids[0],linked_event_ids[1]])
            # no link definitions between corefs
            elif 'coref' in events_doc[event_id] and to_event_id in corefs_doc[events_doc[event_id]['coref']]:
                continue
            elif linked_event_ids not in negatives:
                negatives.append(linked_event_ids)

    #add negatives if not in corefs
    for ind,linked_event_ids in enumerate(negatives):
        if training and ind % 30 != 0:
            continue
        if linked_event_ids not in coref_positives:
            x = build_feature_vector(linked_event_ids,events_doc,corefs_doc)
            X.append(x)
            Y.append(0)
            IDS.append([doc_id,linked_event_ids[0],linked_event_ids[1]])

    for linked_event_ids in coref_positives:
        x = build_feature_vector(linked_event_ids,events_doc,corefs_doc)
        X.append(x)
        Y.append(1)
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

def get_stats(events, corefs, afters, parents,X_train,y_train,IDS):
    number_of_events = sum([len(events[item]) for item in events])
    number_of_docs = len(events)
    number_of_afters = sum([len(afters[item]) for item in afters])
    number_of_candidates = len(y_train)
    number_of_positives = sum(y_train)
    number_of_negatives = number_of_candidates - number_of_positives

    number_of_unique_events = 0

    for doc_id in events:
        number_of_unique_events += len(corefs[doc_id])
        for event_id in events[doc_id]:
            if "coref" not in events[doc_id][event_id]:
                number_of_unique_events +=1

    print("There are %d number of events in %d documents" %(number_of_events,number_of_docs))
    print("There are %d number of unique events" %(number_of_unique_events))
    print("There are %d candidates (%d/%d positive/negative)" %(number_of_candidates,
                                                                number_of_positives, number_of_negatives))
    print("There are %d number of after links" %(number_of_afters))


# filename = "data/LDC2016E130_training.tbf"
def get_dataset(filename, training=True, stats=False):
    ann_file_tbf = os.path.join(PROJECT_FOLDER, filename)
    events, corefs, afters, parents = read_annotations(ann_file_tbf)
    X_train, y_train, IDS = build_feature_matrix_for_dataset(events, corefs, afters, parents,training=training)
    if stats:
        get_stats(events, corefs, afters, parents, X_train, y_train, IDS)
    X_train = preprocess_dataset(X_train)
    return X_train, y_train, IDS, events, corefs, parents
