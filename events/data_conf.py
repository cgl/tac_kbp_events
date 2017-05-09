import os

SOURCE_FOLDER = "data/LDC2017E02/data/2016/eval/eng/nw/source/"
ERE_FOLDER = "data/LDC2017E02/data/2016/eval/eng/nw/ere/"
ANN_FILELIST = os.listdir(ERE_FOLDER)
SOURCE_FILELIST = os.listdir(SOURCE_FOLDER)
ANN_FILENAME = lambda x:  os.path.join(ERE_FOLDER,ANN_FILELIST[x])
SOURCE_FILENAME = lambda x:  os.path.join(SOURCE_FOLDER,SOURCE_FILELIST[x])
