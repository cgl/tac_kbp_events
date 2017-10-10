from collections import defaultdict
import os,pprint,subprocess
from prepare_datafile import Vocabulary, EmbeddingBank
from data_conf import SEQUENCE_SOURCE_FOLDER, PROJECT_FOLDER
from sequence_detection import read_annotations

def get_all_nuggets_from_folders():
    events_all = {}
    folder_list = ["data/LDC2016E130_training.tbf","data/LDC2016E130_test.tbf"]
    for filename in folder_list:
        ann_file_tbf = os.path.join(PROJECT_FOLDER,filename)
        events, _, _, _ = read_annotations(ann_file_tbf)
        events_all.update(events)
    nuggets = set()
    nuggets_dict = defaultdict(int)
    for docs in events_all.values():
        for event_details in docs.values():
            nuggets.add(event_details["nugget"].lower())
            nuggets_dict[event_details["nugget"].lower()] += 1
    print("Number of unique nuggets %d" %len(nuggets))
    #print("\n".join(list(nuggets)))
    #pprint.pprint(nuggets_dict,width=1)
    return nuggets

def update_vocab():
    voc = Vocabulary()
    voc.update_vocab_from_folders()
    voc.write_vocab()

def calculate_cooccurance_table():
    print("hello")
    nuggets = get_all_nuggets_from_folders()
    cooccurence_table = dict()
    for nugget in nuggets:
        cooccurence_table[nugget] = defaultdict(int)

    while nuggets:
        nugget = nuggets.pop()
        for nugget2 in nuggets:
            ps1 = subprocess.Popen(('grep',nugget,'/datasets/EventRegistry/event.registry.docs'), stdout=subprocess.PIPE)
            ps2 = subprocess.Popen(('grep',nugget2 ), stdin=ps1.stdout,stdout=subprocess.PIPE)
            ps1.stdout.close()
            output = subprocess.check_output(('wc', '-l'), stdin=ps2.stdout)
            ps2.wait()
            cooccurence_table[nugget][nugget2] += int(output.strip())

    import ipdb ; ipdb.set_trace()



def update_embeddings():
    emb = EmbeddingBank()
    emb.update_pickle()

def main():
    print("Hello world!")

from optparse import OptionParser
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--main',default=True,action="store_true",help='')
    parser.add_option('-u','--list_nuggets',default=False,action="store_true",help='')
    parser.add_option('-c','--coocurance_table',default=False,action="store_true",help='')

    parser.add_option('--update_vocabulary',default=False,action="store_true",help='')
    parser.add_option('--update_embeddings',default=False,action="store_true",help='')
    (options, args) = parser.parse_args()

    if options.update_vocabulary:
        update_vocab()
    elif options.list_nuggets:
        get_all_nuggets_from_folders()
    elif options.coocurance_table:
        calculate_cooccurance_table()
    elif options.main:
        main()

    if options.update_embeddings:
        update_embeddings()
