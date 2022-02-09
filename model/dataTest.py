import torch
from interCmpnn import *
from tools import *
from torch.utils.data import Dataset, DataLoader
from chemprop.parsing import parse_train_args, modify_train_args

class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self,dataSet,seqContactDict):
        self.dataSet = dataSet
        self.len = len(dataSet)
        self.dict = seqContactDict
        self.properties = [int(x[2]) for x in dataSet]
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles,seq,label = self.dataSet[index]
        contactMap = self.dict[seq]
        return smiles, contactMap, torch.tensor(int(label)),seq

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)

cudanb = '3'
dt = '0712'
randomseed = 4443
testFoldPath = '../data/DUDE/dataPre/DUDE-foldTest1'
trainFoldPath = '../data/DUDE/dataPre/DUDE-foldTrain1'
contactPath = '../data/DUDE/contactMap'
contactDictPath = '../data/DUDE/dataPre/DUDE-contactDict'
time_log('get seq-contact dict....')
seqContactDict = getSeqContactDict(contactPath,contactDictPath)
testProteinList = getTestProteinList(testFoldPath)# whole foldTest
DECOY_PATH = '../data/DUDE/decoy_smile'
ACTIVE_PATH = '../data/DUDE/active_smile'
time_log('get protein-seq dict....')
dataDict = getDataDict(testProteinList,ACTIVE_PATH,DECOY_PATH,contactPath)
time_log('train loader....')
time_log('model args...')
modelArgs = {}
modelArgs['batch_size'] = 1
modelArgs['d_a'] = 32
modelArgs['r'] = 10
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 8
modelArgs['cnn_channels'] = 32
modelArgs['cnn_layers'] = 4
modelArgs['dense_hid'] = 32
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1
#for interCmpnn
modelArgs['hid_dim'] = 32
modelArgs['node_feat_size'] = 39
modelArgs['edge_feat_size'] = 10
modelArgs['graph_feat_size'] = 32
modelArgs['num_layers'] = 2
modelArgs['num_timesteps'] = 2
modelArgs['n_layers'] = 1
modelArgs['n_heads'] = 8
modelArgs['pf_dim'] = 200
mpnargs = parse_train_args()
mpnargs.data_path = '../data/bbbp.csv'
mpnargs.dataset_type = 'classification' # regression
mpnargs.num_folds = 5
mpnargs.gpu = 0
mpnargs.epochs = 30
mpnargs.ensemble_size = 1
mpnargs.batch_size = 32
mpnargs.hidden_size = 32
mpnargs.split_sizes = [0.7, 0.1, 0.2] # 5-fold cross-validation
mpnargs.seed =521
mpnargs.depth = 4
modify_train_args(mpnargs)

#for train
time_log('train args...')
trainArgs = {}
trainArgs['model'] = MPN(mpnargs,None,None,False,modelArgs).cuda()
trainArgs['epochs'] = 1
trainArgs['lr'] = 0.0001
trainArgs['doTest'] = True
trainArgs['test_proteins'] = testProteinList
trainArgs['testDataDict'] = dataDict
trainArgs['seqContactDict'] = seqContactDict
trainArgs['use_regularizer'] = False
trainArgs['penal_coeff'] = 0.03
trainArgs['clip'] = True
trainArgs['criterion'] = torch.nn.BCELoss()

time_log('train args over...')
