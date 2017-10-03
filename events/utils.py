import os
import source_parser as html_parser
from prepare_datafile import Vocabulary, EmbeddingBank
from data_conf import SEQUENCE_SOURCE_FOLDER

def get_all_text_from_folders(folder_list):
    my_parser = html_parser.MyHTMLParser()
    for folder in folder_list:
        list_dir = os.listdir(folder)
        for filename in list_dir:
            if filename.endswith("txt"):
                with open(os.path.join(folder,filename)) as sourcefile:
                    source = sourcefile.read()
                    my_parser.feed(source)
    return my_parser.get_text()

def update_vocab_from_folders():
    voc = Vocabulary()
    training_folder = os.path.join(SEQUENCE_SOURCE_FOLDER,"training")
    test_folder = os.path.join(SEQUENCE_SOURCE_FOLDER,"test")
    folder_list = [training_folder,test_folder]
    text = get_all_text_from_folders(folder_list)
    voc.update_vocab_from_text(text)
    voc.write_vocab()

def update_embeddings():
    emb = EmbeddingBank()
    emb.update_pickle()

def main():
    print("Hello world!")

from optparse import OptionParser
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--main',default=True,action="store_true",help='')
    parser.add_option('--update_vocabulary',default=False,action="store_true",help='')
    parser.add_option('--update_embeddings',default=False,action="store_true",help='')
    (options, args) = parser.parse_args()

    if options.update_vocabulary:
        update_vocab_from_folders()
    elif options.main:
        main()

    if options.update_embeddings:
        update_embeddings()
