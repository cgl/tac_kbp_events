import logging,os,sys,pickle,random
import numpy as np
from xml.etree import ElementTree
from data_conf import *
#check the python version to Get User Input From the Command Line
from sys import version_info
from nltk import word_tokenize, sent_tokenize
from collections import defaultdict
py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2
if not py3:
    input = raw_input

# source: https://github.com/anoperson/jointEE-NN/blob/master/jee_processData.py#L353
# '../data/GoogleNews-vectors-negative300.bin'
def load_bin_vec(vocab,fname='/datasets/GoogleNews-vectors-negative300.bin'):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = {}
    dim = 0
    for word in vocab:
        try:
            word_vecs[word] = model.word_vec(word)
            dim = word_vecs[word].shape[0]
        except KeyError as e:
            logging.warning(e)
    print('dim: ', dim)
    return dim, word_vecs

# source: https://github.com/anoperson/jointEE-NN/blob/master/jee_processData.py#L338
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def spans(txt):
    tokens=word_tokenize(txt)
    offset = 0
    for token in tokens:
        if token not in ["''",'``']:
            offset = txt.find(token, offset)
            yield token, offset, offset+len(token)
            offset += len(token)

def tokenize_with_span(source_str):
    for token in spans(source_str):
        yield token
        assert token[0]==source_str[token[1]:token[2]]

class Vocabulary(object):
    vocab = defaultdict(float)
    vocab_filename = os.path.join(PROJECT_FOLDER,"data/vocab.txt")

    def write_vocab(self):
        with open(self.vocab_filename,"w") as v_file:
            for key, value in self.vocab.items():
                v_file.write("%s\t%s\n" % (key,value))

    def read_vocab(self):
        self.vocab = []
        try:
            with open(self.vocab_filename,"r") as v_file:
                for line in iter(v_file.readline, ''):
                    [word,_] = line.split("\t")
                    self.vocab.append(word)
        except FileNotFoundError as e:
            logging.warning(e)

    def update_vocab(self,source_filename):
        tree = ElementTree.parse(source_filename)
        root = tree.getroot()
        for child in root.getchildren():
            #print(child.text.strip())
            for par in child.text.strip().split("\n"):
                if par:
                    sentences = sent_tokenize(par)
                    for sent in sentences:
                        words = word_tokenize(sent)
                        for word in words:
                            self.vocab[word] += 1
    def update_vocab_from_text(self,text):
        if text:
            for par in text.split("\n"):
                if par:
                    sentences = sent_tokenize(par)
                    for sent in sentences:
                        words = word_tokenize(sent)
                        for word in words:
                            self.vocab[word] += 1

class EmbeddingBank():
    vocab_obj = Vocabulary()
    vector_file = '/datasets/GoogleNews-vectors-negative300.bin'

    def __init__(self,vector_file=None,vocab_file=None):
        self.W_fname = "../data/vectors.pickle"
        if os.path.isfile(self.W_fname):
            pickled = pickle.load(open(self.W_fname,"rb"))
            self.W = pickled['W']
            self.word_idx_map = pickled['W_ind']
        else:
            if vector_file:
                self.vector_file = vector_file
            if vocab_file:
                self.vocab_obj.vocab_filename = vocab_file
            self.update_pickle()

    def update_pickle(self):
        self.vocab_obj.read_vocab()
        self.calculate_vector_list()
        pickle.dump({'W': self.W , 'W_ind' : self.word_idx_map} , open(self.W_fname,"wb"))

    def calculate_vector_list(self):
        dim, self.word_vecs = load_bin_vec(self.vocab_obj.vocab) # fname=FLAGS.w2v_file
        print("Loading idx map...")
        self.W, self.word_idx_map = get_W(self.word_vecs)

    def get_index(self,word):
        if word in self.word_idx_map:
            ind = self.word_idx_map.get(word)
        else:
            ind = 0
        return ind

    def get_embedding(self,word):
        ind = self.get_index(word)
        return self.W[ind]


class Dataset(object):
    project_folder = PROJECT_FOLDER
    training_source_folder = os.path.join(project_folder,"data/LDC2017E02/data/2016/eval/eng/nw/source/")
    training_ann_folder = os.path.join(project_folder,"data/LDC2017E02/data/2016/eval/eng/nw/ere/")
    test_source_folder = os.path.join(project_folder,"data/LDC2017E02/data/2016/eval/eng/df/source/")
    test_ann_folder = os.path.join(project_folder,"data/LDC2017E02/data/2016/eval/eng/df/ere/")
    training_dataset_file = os.path.join(project_folder,"data/dataset_file_training.txt")
    test_dataset_file =  os.path.join(project_folder,"data/dataset_file_test.txt")
    vocab_obj = Vocabulary()
    vocab = vocab_obj.vocab
    vocab_filename = vocab_obj.vocab_filename
    training_set, test_set = None, None
    label_set = ['None','broadcast', 'injure', 'transportperson', 'transfermoney', 'artifact', 'contact',
                 'elect', 'correspondence', 'startposition', 'transportartifact', 'demonstrate', 'arrestjail',
                 'meet', 'transferownership', 'transaction', 'die', 'attack', 'endposition']
    window_size = 3
    def __init__(self, vocab_filename=None,training_dataset_file=None,test_dataset_file=None):
        if vocab_filename:
            vocab_obf.vocab_filename = vocab_filename
            vocab_obj.read_vocab()
        if training_dataset_file:
            self.training_dataset_file = training_dataset_file
        if test_dataset_file:
            self.test_dataset_file = test_dataset_file

    def process(self,dataset_files_exist=True):
        if not dataset_files_exist:
            self.prepare_dataset_file(self.training_dataset_file,self.training_source_folder, self.training_ann_folder)
            self.prepare_dataset_file(self.test_dataset_file,self.test_source_folder, self.test_ann_folder)
            vocab_obj.write_vocab()
        else:
            vocab_obj.read_vocab()
        self.build_dataset()
        self.show_label_percentage()

    def build_dataset(self):
        self.training_set = self.load_data_and_labels(self.training_dataset_file)
        self.test_set = self.load_data_and_labels(self.test_dataset_file)

    def set_training_folders(self,training_source_folder, training_ann_folder):
        self.training_source_folder, self.training_ann_folder = training_source_folder, training_ann_folder

    def set_test_folders(self,test_source_folder, test_ann_folder):
        self.test_source_folder, self.test_ann_folder = test_sourse_folder, test_ann_folder,


    # <span style="color: red">{}</span>
    def process_datafile(self,ann_filename,source_filename,dataset_file):
        with open(source_filename) as input:
            source_str = input.read()
        vocab.update_vocab(source_filename)
        tree = ElementTree.parse(ann_filename)
        root = tree.getroot()
        offsets = {}
        hoppers = root.find("hoppers")
        for hopper in hoppers:
            event_mentions = hopper.getchildren()
            for em in event_mentions:
                trigger = em.find("trigger")
                id = em.get("id")
                subtype = em.get('subtype') #em.get("type")
                offsets[trigger.get('offset')] = (trigger.get('length'),id,subtype)
        # offsets have event trigger offsets, length and id

        tokenized_source = tokenize_with_span(source_str)
        with open(dataset_file,"a") as output:
            for token_span in tokenized_source:
                subtype = None
                word = token_span[0]
                if word in self.vocab:
                    if str(token_span[1]) in offsets.keys():
                        subtype = offsets.pop(str(token_span[1]))[2]
                    print("%s\t%s\t%s\t%s" %(word,token_span[1],token_span[2],subtype), file=output)

    # writes dataset to file
    def prepare_dataset_file(self,dataset_file,
                             source_folder=os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/df/source/"),
                             ere_folder = os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/df/ere/"),
                             append=False):
        """
        Takes source folder, annotation folder and dataset file to write the data as arguments.
        Reads dataset from source and annotation folder and writes the data to dataset file.
        """
        if not append:
            try:
                os.remove(dataset_file)
            except FileNotFoundError:
                pass
        file_index = 0
        ann_filelist = os.listdir(ere_folder)
        source_filelist = os.listdir(source_folder)
        ann_filename_fun = lambda x:  os.path.join(ere_folder,ann_filelist[x])
        source_filename_fun = lambda x:  os.path.join(source_folder,source_filelist[x])

        list_dir = os.listdir(source_folder)
        while file_index < len(list_dir):
            ann_filename = ann_filename_fun(file_index)
            source_filename = source_filename_fun(file_index)
            self.process_datafile(ann_filename,source_filename,dataset_file)
            file_index += 1


    def load_data_and_labels(self,dataset_file):
        dataset = []
        x,y_text = [],[]
        with open(dataset_file,"r") as df:
            for line in iter(df.readline, ''):
                [word,_,_,subtype] = line.split("\t")
                if word in self.vocab: # todo should fix unknown words
                    dataset.append((word,subtype.strip()))
        for i in range(len(dataset)):
            line = dataset[i]
            if line[1] != 'None' or random.random() < 0.002:
                x.append([a[0] if a[0] in self.vocab else "" for a in dataset[i-self.window_size:i+self.window_size+1]])
                #x.append(line[0])
                y_text.append(line[1])

        y = [self.label_set.index(item) for item in y_text]

        #for i in range(len(y)):
        #  assert one_hot[y[i]].all() == one_hot_y[i].all()
        return x,y,dataset

    def show_label_percentage(self):
        training = self.training_set
        test = self.test_set
        print("Tr #\tTr %\tTe #\tTe %\tlabel\n")
        for i in range(0,len(self.label_set)):
            print("%d\t%.2f\t%d\t%.2f\t%s" %(training[1].count(i),training[1].count(i)/len(training[1]),
                                             test[1].count(i),test[1].count(i)/len(test[1]),
                                             self.label_set[i]))

def get_one_hot(y,label_count):
        #one_hot = {}
        #for i in range(len(set(y))):
        #  one_hot[set_y[i]] = np.zeros(len(set_y))
        #  one_hot[set_y[i]][i] = 1
        #  one_hot_y = [one_hot[item] for item in y]
        identity = np.identity(label_count)
        one_hot_y = [identity[item] for item in y]
        return one_hot_y


def load_vocab():
    vocab = []
    with open(VOCABFILE,"r") as datafile:
      for line in iter(datafile.readline, ''):
        [word,_] = line.split("\t")
        vocab.append(word)
    return vocab

# from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py#L48
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

def embeddings():
  filename = 'glove.6B.50d.txt'
  vocab,embd = loadGloVe(filename)
  vocab_size = len(vocab)
  embedding_dim = len(embd[0])
  embedding = np.asarray(embd)


def initialize(): # todo delete
    """
    Creates datafile and vocab file
    """
    file_index=0
    list_dir = os.listdir(SOURCE_FOLDER)

    while file_index < len(list_dir):
        ann_filename = ANN_FILENAME(file_index)
        source_filename = SOURCE_FILENAME(file_index)
        prepare_datafile(ann_filename,source_filename,DATAFILE)
        file_index += 1

    """
    dimEmb, w2v = load_bin_vec(w2v_file, vocab)
    W1, word_idx_map = get_W(w2v, dimEmb)

    dictionaries = {}
    dictionaries['word'] = word_idx_map
    embeddings = {}
    embeddings['word'] = W1
    """
    #print vocabulary to file
    with open(VOCABFILE,"w") as v_file:
      for key, value in vocab.items():
        v_file.write("%s\t%s\n" % (key,value))


if __name__ == "__main__":
    response = input("Do you want to change the data folder? (default:%s) [y/N]" % SOURCE_FOLDER)
    if response is "y":
        response = input("Enter the directory name of your source files: ")
        SOURCE_FOLDER = response
    response = input("Do you want to list folder content? [y/N]: ")
    if response is "y":
        sys.stdout.write("From the above files,")
        print("".join(["%d. %s" %(ind,filename) for ind,filename in enumerate(list_dir)]))
    initialize()
