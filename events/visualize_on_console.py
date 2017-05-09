import os,sys
from xml.etree import ElementTree
from data_conf import *
#check the python version to Get User Input From the Command Line
from sys import version_info
py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2
if not py3:
  input = raw_input

# <span style="color: red">{}</span>
def visualise_file(ann_filename,source_filename):
    with open(source_filename) as input:
        source = input.read()
    tree = ElementTree.parse(ann_filename)
    root = tree.getroot()
    offsets = []
    for trigger in root.findall("hoppers/hopper/event_mention/trigger"):
        offsets.append((trigger.get('offset'),trigger.get('length')))
    sorted_offsets = sorted(offsets,key=lambda x: int(x[0]))
    print(sorted_offsets)
    for index in reversed(range(0,len(sorted_offsets))):
        offset,length = sorted_offsets[index]
        offset = int(offset)
        length = int(length)
        source = source[:offset] + "\033[44;37m[%s]" %index + source[offset:offset+length] + "\033[m" + source[offset+length:]
    sys.stdout.write(source.replace("\n\n","\n"))

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
    results = []
    while True:
        file_index = input("Please enter the number of file to start [1-%d]: " %len(list_dir))
        if file_index.isnumeric():
            file_index = int(file_index)-1
            break
        else:
            print("\n".join(["%d. %s" %(ind,filename) for ind,filename in enumerate(list_dir)][0:10]))
            sys.stderr.write("Index should be an integer!\n")

    while file_index <= len(list_dir):
        ann_filename = ANN_FILENAME(file_index)
        source_filename = SOURCE_FILENAME(file_index)
        visualise_file(ann_filename,source_filename)
        response = input("Please enter the index of main event (q for quit): ")
        if response is "q":
            break
        results.append((response,ann_filename))
        file_index += 1
    print(results)
