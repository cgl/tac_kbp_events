import os

PROJECT_FOLDER=os.path.abspath(os.path.join(os.path.abspath(os.curdir), os.pardir))
DATAFILE = os.path.join(PROJECT_FOLDER,"data/datafile.txt")
VOCABFILE = os.path.join(PROJECT_FOLDER,"data/vocab.txt")

SOURCE_FOLDER = os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/nw/source/")
ERE_FOLDER = os.path.join(PROJECT_FOLDER,"data/LDC2017E02/data/2016/eval/eng/nw/ere/")
try::
    ANN_FILELIST = os.listdir(ERE_FOLDER)
    SOURCE_FILELIST = os.listdir(SOURCE_FOLDER)
    ANN_FILENAME = lambda x:  os.path.join(ERE_FOLDER,ANN_FILELIST[x])
    SOURCE_FILENAME = lambda x:  os.path.join(SOURCE_FOLDER,SOURCE_FILELIST[x])
except Exception as e:
    print(e.args[0])

event_type_index = {'Contact_Meet':0 ,
                    'Movement_Transport-Artifact':1 ,
                    'Life_Divorce':2 ,
                    'Justice_Extradite':3 ,
                    'Life_Marry':4 ,
                    'Life_Injure':5 ,
                    'Conflict_Attack':6 ,
                    'Life_Be-Born':7 ,
                    'Conflict_Demonstrate':8 ,
                    'Transaction_Transfer-Money':9 ,
                    'Justice_Pardon':10 ,
                    'Personnel_Nominate':11 ,
                    'Transaction_Transfer-Ownership':12 ,
                    'Justice_Appeal':13 ,
                    'Justice_Fine':14 ,
                    'Business_Declare-Bankruptcy':15 ,
                    'Contact_Broadcast':16 ,
                    'Business_Start-Org':17 ,
                    'Life_Die':18 ,
                    'Justice_Arrest-Jail':19 ,
                    'Justice_Acquit':20 ,
                    'Contact_Correspondence':21 ,
                    'Justice_Execute':22 ,
                    'Justice_Charge-Indict':23 ,
                    'Business_Merge-Org':24 ,
                    'Contact_Contact':25 ,
                    'Personnel_Start-Position':26 ,
                    'Manufacture_Artifact':27 ,
                    'Personnel_End-Position':28 ,
                    'Movement_Transport-Person':29 ,
                    'Justice_Sentence':30 ,
                    'Justice_Release-Parole':31 ,
                    'Justice_Sue':32 ,
                    'Personnel_Elect':33 ,
                    'Justice_Convict':34 ,
                    'Justice_Trial-Hearing':35,
                    'Business_End-Org' : 36,
                    #manual entries:
                    'Transaction_Transaction': 37,
                    'Transaction' : 38,
}

realis_index = {'Other':1,
                'Actual':2,
                'Generic':3,
                'NOT_ANNOTATED' : 0,
}
