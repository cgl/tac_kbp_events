import os,sys
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
vocab = defaultdict(float)

# source: https://github.com/anoperson/jointEE-NN/blob/master/jee_processData.py#L353
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               dim = word_vecs[word].shape[0]
            else:
                f.read(binary_len)
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

# <span style="color: red">{}</span>
def prepare_datafile(ann_filename,source_filename,datafile):
    with open(source_filename) as input:
        source_str = input.read()
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

    with open(datafile,"a") as output:
      for token_span in tokenized_source:
        subtype = None
        if str(token_span[1]) in offsets.keys():
          subtype = offsets.pop(str(token_span[1]))[2]
        word = token_span[0]
        print("%s\t%s\t%s\t%s" %(word,token_span[1],token_span[2],subtype), file=output)
        vocab[word] += 1
    return vocab

def load_data_and_labels():
  x,y = [],[]
  with open(DATAFILE,"r") as datafile:
    for line in iter(datafile.readline, ''):
      [word,_,_,subtype] = line.split("\t")
      x.append(word)
      y.append(subtype.strip())
  return x,y

def load_vocab():
  vocab = []
  with open(VOCABFILE,"r") as datafile:
    for line in iter(datafile.readline, ''):
      [word,_] = line.split("\t")
      vocab.append(word)
  return vocab


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


def initialize():
    """
    Creates datafile and vocab file
    """
    results = []
    file_index=0
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
    list_dir = os.listdir(SOURCE_FOLDER)
    if response is "y":
        sys.stdout.write("From the above files,")
        print("".join(["%d. %s" %(ind,filename) for ind,filename in enumerate(list_dir)]))
    initialize()
