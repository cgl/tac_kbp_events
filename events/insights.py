from optparse import OptionParser
from sequence_detection import get_dataset

def preview_nuggets(stats=False):
    X_train,y_train,IDS,events_train = get_dataset("data/LDC2016E130_training.tbf",stats=stats,training=True)
    X_test,y_test,IDS_test,events_test = get_dataset("data/LDC2016E130_test.tbf",stats=stats,training=False)
    for ind,(doc_id,e1_id,e2_id) in enumerate(IDS):
        print("%s\t%s\t" %(events_train[doc_id][e1_id]['nugget'], events_train[doc_id][e2_id]['nugget']))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-s','--statistics',default=False,action="store_true",help='')
    (options, args) = parser.parse_args()
    if options.statistics:
        pass
    else:
        preview_nuggets()
