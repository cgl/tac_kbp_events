from collections import defaultdict
from optparse import OptionParser
from sequence_detection import get_dataset

def preview_nuggets(stats=False):
    X_train,y_train,IDS,events = get_dataset("data/LDC2016E130_training.tbf",stats=stats,training=True)
    X_test,y_test,IDS_test,events_test = get_dataset("data/LDC2016E130_test.tbf",stats=stats,training=True)
    IDS.extend(IDS_test)
    events.update(events_test)
    Y = y_train + y_test
    for ind,(doc_id,e1_id,e2_id) in enumerate(IDS):
        if Y[ind]:
            e1_offset= events[doc_id][e1_id]['offsets'].split(",")[0]
            e2_offset= events[doc_id][e2_id]['offsets'].split(",")[0]
            print("\t".join([doc_id,e1_id,e2_id,e1_offset,e2_offset,
                             str(int(e1_offset)-int(e2_offset)),
            ]))

    import ipdb ; ipdb.set_trace()

def print_nugget_pairs(events,IDS,Y):
    from_to = defaultdict(list)
    for ind,(doc_id,e1_id,e2_id) in enumerate(IDS):
        if Y[ind]:
            e1_nugget= events[doc_id][e1_id]['nugget'].lower()
            e2_nugget= events[doc_id][e2_id]['nugget'].lower()
            #print("%s\t%s\t" %(e1_nugget, e2_nugget))
            from_to[e1_nugget].append(e2_nugget)
    #print("\n".join([key+"\t"+" ".join(item) for key,item in from_to.items() if len(set(item)) != len(item)]))
    for key,value in from_to.items():
        to_list = []
        for to_nugget in set(value):
            to_list.append("%s%s" %(to_nugget,"(%s)" %value.count(to_nugget) if value.count(to_nugget) > 1 else ""))
        print("%s\t%s" %(key,", ".join(to_list)))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-s','--statistics',default=False,action="store_true",help='')
    (options, args) = parser.parse_args()
    if options.statistics:
        pass
    else:
        preview_nuggets()
