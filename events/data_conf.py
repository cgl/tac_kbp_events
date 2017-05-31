import os

PROJECT_FOLDER=os.path.abspath(os.path.join(os.path.abspath(os.curdir), os.pardir))
SOURCE_FOLDER = os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/nw/source/")
ERE_FOLDER = os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/nw/ere/")
ANN_FILELIST = os.listdir(ERE_FOLDER)
SOURCE_FILELIST = os.listdir(SOURCE_FOLDER)
ANN_FILENAME = lambda x:  os.path.join(ERE_FOLDER,ANN_FILELIST[x])
SOURCE_FILENAME = lambda x:  os.path.join(SOURCE_FOLDER,SOURCE_FILELIST[x])
DATAFILE = os.path.join(PROJECT_FOLDER,"data/datafile.txt")
VOCABFILE = os.path.join(PROJECT_FOLDER,"data/vocab.txt")
