from tools import *
from data import *
from sklearn import metrics
def train(trainArgs,mpnargs):
    """
    args:
        model           : {object} model
        lr              : {float} learning rate
        train_loader    : {DataLoader} training data loaded into a dataloader
        doTest          : {bool} do test or not
        test_proteins   : {list} proteins list for test
        testDataDict    : {dict} test data dict
        seqContactDict  : {dict} seq-contact dict
        optimizer       : optimizer
        criterion       : loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
        epochs          : {int} number of epochs
        use_regularizer : {bool} use penalization or not
        penal_coeff     : {int} penalization coeff
        clip            : {bool} use gradient clipping or not
    Returns:
        accuracy and losses of the model
    """
    losses = []
    accs = []
    testResults = {}
    for i in range(trainArgs['epochs']):
        time_log("Running_EPOCH:%s"%(i+1))
        total_loss = 0
        n_batches = 0
        correct = 0
        wres = []
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']
        attention_model.train()
        for batch_idx,(lines, contactmap,properties,seq) in enumerate(train_loader):  
            input, y = make_variables_s(lines, properties,mpnargs)
            contactmap = create_variable(contactmap[0])
            y_pred,att = attention_model(input,None,contactmap)
            y_pred = y_pred.squeeze(0)
            if trainArgs['use_regularizer']:
                print('use_regularizer')
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,att.size(1),att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)
            if not bool(attention_model.task_type) :
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor).unsqueeze(1)).data.sum()
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))+(trainArgs['penal_coeff'] * penal.cpu()/train_loader.batch_size)
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            optimizer.zero_grad()
            loss.backward()
            total_loss+=loss.item()
            #gradient clipping
            if trainArgs['clip']:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
            time_log("Epoch-%s_batch-%s"%(i+1,batch_idx))
            wres.append('%s,%s,%s,%s,%s,%s\n'%(i+1,batch_idx,lines[0],seq[0],properties[0].cpu().tolist(),y_pred.squeeze(0).cpu().tolist()[0]))
            torch.cuda.empty_cache()
        with open('../myres/%strain.csv'%(trainArgs['saveNamePre']),'a+') as f:
            f.writelines(wres)
        avg_loss = total_loss/n_batches
        acc = correct.numpy()/(len(train_loader.dataset))
        
        losses.append(avg_loss)
        accs.append(acc)
        
        time_log("avg_loss is%s"%avg_loss)
        time_log("train ACC = %s"%acc)
        modelfile= '../model_pkl/my/'+trainArgs['saveNamePre']+'%d.pkl'%(i+1)
        if(trainArgs['doSave']):
            torch.save(attention_model.state_dict(), modelfile)
        if(trainArgs['doTest']):
            testArgs = {}
            testArgs['model'] = attention_model
            testArgs['test_proteins'] = trainArgs['test_proteins']
            testArgs['seqContactDict'] = trainArgs['seqContactDict']
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']

            testResult = testPerProtein(testArgs)
            testResults[i] = testResult
        torch.cuda.empty_cache()
    return losses,accs,testResults
def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce
def testPerProtein(testArgs):
    result = {}
    for x in testArgs['test_proteins']:
        time_log('\n current test protein:%s'%(x.split('_')[0]))
        data = getTrainDataSet(testFoldPath)
        test_dataset = ProDataset(dataSet = data,seqContactDict = testArgs['seqContactDict'])
        test_loader = DataLoader(dataset=test_dataset,batch_size=1,collate_fn=my_collate, pin_memory=True, shuffle=True,drop_last = True)
        testArgs['test_loader'] = test_loader
        testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4 = test(testArgs)
        result[x] = [testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4]
    return result
def test(testArgs):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    time_log('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    attention_model.eval()
    with torch.no_grad():
        for batch_idx,(lines, contactmap,properties,seq) in enumerate(test_loader):
            input, y = make_variables_s(lines, properties,mpnargs)
            contactmap = create_variable(contactmap[0])
            y_pred,att = attention_model(input,None,contactmap)
            y_pred = y_pred.squeeze(0)
            if not bool(attention_model.task_type):
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
                all_pred=np.concatenate((all_pred,y_pred.data.cpu().squeeze(1).numpy()),axis = 0)
                all_target = np.concatenate((all_target,y.data.cpu().numpy()),axis = 0)
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))+(C * penal.cpu()/train_loader.batch_size)
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            n_batches+=1
            time_log('%s,%s,%s,%s,%s'%(batch_idx,lines[0],seq[0],properties[0].cpu().tolist(),y_pred.squeeze(0).cpu().tolist()[0]))
    testSize = round(len(test_loader.dataset),3)
    testAcc = round(correct.numpy()/(n_batches*test_loader.batch_size),3)
    testRecall = round(metrics.recall_score(all_target,np.round(all_pred)),3)
    testPrecision = round(metrics.precision_score(all_target,np.round(all_pred)),3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred),3)
    time_log("AUPR =  %s"%metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item()/n_batches,5)
    time_log("test size = %s  test acc = %s  test recall = %s  test precision = %s  test auc = %s  test loss = %s"%(testSize,testAcc,testRecall,testPrecision,testAuc,testLoss))
    roce1 = round(getROCE(all_pred,all_target,0.5),2)
    roce2 = round(getROCE(all_pred,all_target,1),2)
    roce3 = round(getROCE(all_pred,all_target,2),2)
    roce4 = round(getROCE(all_pred,all_target,5),2)
    time_log("roce0.5 =%s  roce1.0 =%s  roce2.0 =%s  roce5.0 =%s"%(roce1,roce2,roce3,roce4))
    return testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4
