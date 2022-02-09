import os
import sys
import random
import torch
import warnings
from modelUtilsTest import *
import socket
warnings.filterwarnings("ignore")
def seed_torch(seed=521):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
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
print('testFoldPath:%s'%testFoldPath)
def log_pam(ags,model = 0 ):
    longlist= ['test_proteins','testDataDict','seqContactDict','model','train_loader']
    if model == 0:
        print(['%s:%s'%(k,v) for k,v in ags.items() if k not in longlist])
    else:
        print(['%s:%s'%(k,vars(ags)[k]) for k in  list(vars(ags).keys()) if k not in longlist])
log_pam(modelArgs)
log_pam(trainArgs)
log_pam(mpnargs,1)

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SEAL-CI model.
    """
    testResults = train(trainArgs,mpnargs)
    time_log('AD')
if __name__ == "__main__":
    main()