import os
from xml.etree import ElementTree

SOURCE_FOLDER = "data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2016/eval/eng/nw/source/"
ERE_FOLDER = "data/LDC2017E02_TAC_KBP_Event_Nugget_Detection_and_Coreference_Comprehensive_Training_and_Evaluation_Data_2014-2016/data/2016/eval/eng/nw/ere/"
ANN_FILELIST = os.listdir(ERE_FOLDER)
SOURCE_FILELIST = os.listdir(SOURCE_FOLDER)
ann_filename = os.path.join(ERE_FOLDER,ANN_FILELIST[0])
source_filename = os.path.join(SOURCE_FOLDER,SOURCE_FILELIST[0])

# <span style="color: red">{}</span>
def visualise_file(ann_filename,source_filename):
    with open(source_filename) as input:
        source = input.read()
    tree = ElementTree.parse(ann_filename)
    root = tree.getroot()
    offsets = []
    for trigger in root.findall("hoppers/hopper/event_mention/trigger"):
        offsets.append((trigger.get('offset'),trigger.get('length')))
    offsets.reverse()
    for offset,length in offsets:
        offset= int(offset)
        length = int(length)
        source = source[:offset] + "\033[44;33m" + source[offset:offset+length] + "\033[m" + source[offset+length:]
    print(source)

if __name__ == "__main__":

    visualise_file(ann_filename,source_filename)
