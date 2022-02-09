import os
import torch
import random
from sklearn import metrics
import warnings
from modelUtils import *
import socket
warnings.filterwarnings("ignore")
def seed_torch(seed=521):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    print('seed:',seed)
seed_torch(randomseed)
os.environ['CUDA_VISIBLE_DEVICES']=cudanb
print('current pid', os.getpid())
print('host:',socket.gethostname())
print('cuda:',os.environ['CUDA_VISIBLE_DEVICES'])
print('trainFoldPath:%s'%trainFoldPath)
print('testFoldPath:%s'%trainFoldPath)
def log_pam(ags):
    longlist= ['test_proteins','testDataDict','seqContactDict','model','train_loader']
    print(['%s:%s'%(k,v) for k,v in ags.items() if k not in longlist])
log_pam(modelArgs)
log_pam(trainArgs)
def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SEAL-CI model.
    """
    losses,accs,testResults = train(trainArgs,mpnargs)
    
if __name__ == "__main__":
    main()