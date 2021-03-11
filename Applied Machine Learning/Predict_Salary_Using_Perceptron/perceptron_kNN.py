import numpy as np
import time

# define function to get mapping
def getMapping(trainingData):
    data_list = [list(tup) for tup in trainingData]

    mapping = {}
    reMap = {}

    for row in data_list:
        for i, j in enumerate(row):
            feature = (i, j)
            if feature not in mapping:
                reMap[len(mapping)] = feature
                mapping[feature] = len(mapping) # insert a new feature in the index
    return mapping, reMap

# define one-hot encoder
def oneHotEncoding(dataset, mapping):
    data_list = [list(x) for x in dataset]
    newData = []
    
    for row in data_list:
        newRow = []
        for i, j in enumerate(row):
            feature = (i, j)
            if feature not in mapping:
                continue
            newRow.append(mapping[feature])
        newData.append(newRow)
    
    binaryData=np.zeros((len(newData), len(mapping)))
    for i, row in enumerate(newData):
        for x in row:
            binaryData[i][x] = 1
    return binaryData 

# define fuction to predict outcome
def get_predict(oneHotTest,w):
    return np.sign(oneHotTest.dot(w))

# define function to error and positive rate
def get_error_rate(predict, oneHotDev):
    errorRate = round(sum(predict != oneHotDev[:,-1])*100.0/len(oneHotDev),2)
    posRate = round(sum(predict == 1)*100.0/len(oneHotDev),2)
    return errorRate, posRate

# defien average perceptron (smart version)
def perc_avg_smart(oneHotTrain,oneHotDev, epochMax=5, w0=0, printErr=True):
    devErrList = []
    w = np.zeros(len(oneHotTrain[0])-1)  # [0,0]
    w[0] = w0
    wAvg = np.zeros(len(oneHotTrain[0])-1)  # [0,0]
    wFinal = np.zeros((epochMax, len(oneHotTrain[0])-1))
    c = 0
    epoch, total, updates = 0, 0, 1
    while ((updates > 0) & (epoch < epochMax)):
        epoch += 1
        updates = 0           
        for i, x in enumerate(oneHotTrain[:,:-1]):
            y = oneHotTrain[i][-1]
            if x.dot(w)*y <= 0:
                w += y*x
                wAvg += c*(y*x)
                updates += 1
            c += 1

        wFinal[epoch-1] = c*w - wAvg
        
        # get dev error rate
        devPredict = get_predict(oneHotDev[:,:-1], wFinal[epoch-1])
        devErrorRate, devPositive = get_error_rate(devPredict, oneHotDev)
        devErrList.append(devErrorRate)
        
        # print updates and error rate
        if printErr:
            print "epoch %d, updates %d (%s%%), dev error %s%% (+:%s%%)" % (epoch, updates, 
                                                                    str(updates*100.0/len(oneHotTrain)), 
                                                                    str(devErrorRate),str(devPositive))
    return wFinal, np.array(devErrList)

# function to get predicted outcomes based on Euclidean distance in k nearest neighbour algorithm
def eucDist(trainSet, testSet, k):
    predictDev=np.zeros(len(testSet))
    for indx, row in enumerate(testSet):
        eucDist = np.linalg.norm(trainSet[:,:-1]-row, axis=1)
        if k < len(eucDist):
            topKIndx = np.argpartition(eucDist, k)[:k]
        else:
            topKIndx = np.argpartition(eucDist, len(eucDist)-1)
        if trainSet[:,-1][topKIndx].mean() > 0.5:
            predictDev[indx] = 1
    return predictDev

# Load data
train = np.genfromtxt('income.train.txt.5k', delimiter=',', 
                       dtype={'names':('age', 'sector', 'education', 'maritalStatus', 
                                       'occupation', 'race', 'gender', 'workHours', 'country', 
                                       'income'), 
                              'formats':('i4', 'U30','U30','U30', 'U30','U30', 'U30', 'i4', 
                                         'U30','U30')})

dev =  np.genfromtxt('income.dev.txt', delimiter=',',
                     dtype={'names':('age', 'sector', 'education', 'maritalStatus',
                                     'occupation', 'race', 'gender', 'workHours', 'country',
                                     'income'),
                            'formats':('i4', 'U30','U30','U30', 'U30','U30', 'U30', 'i4',
                                       'U30','U30')})

# get feature map and reverse map
fields = ['education', 'occupation']
featureMap, featureReMap = getMapping(train[fields])

# get one-hot encoded train, dev and test sets
oneHotTrain = oneHotEncoding(train[fields], featureMap)
oneHotDev = oneHotEncoding(dev[fields], featureMap)

oneHotTrainKnn = np.column_stack((oneHotTrain, 
                               np.array([0 if x == ' <=50K' else 1 for x in train['income']])))
oneHotDevKnn = np.column_stack((oneHotDev,
                            np.array([0 if x == ' <=50K' else 1 for x in dev['income']])))

oneHotTrain = np.column_stack((np.ones(len(oneHotTrain)), oneHotTrain, 
                               np.array([-1 if x == ' <=50K' else 1 for x in train['income']])))
oneHotDev = np.column_stack((np.ones(len(oneHotDev)), oneHotDev,
                            np.array([-1 if x == ' <=50K' else 1 for x in dev['income']])))


# error and positive rate on dev set: Euclidean distance
kValues = [1,5,9, 59, 99]
rateDevEuc = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time()    
    predictDev=eucDist(oneHotTrainKnn, oneHotDevKnn[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDevKnn[:,-1])*100.00/len(oneHotDevKnn)
    positiveRate = sum(predictDev ==1)*100.00/len(oneHotDevKnn)
    rateDevEuc[i] = (k,errorRate,positiveRate,time_)

print('\nk-values, predicted error rate and positive rate on dev set (k-NN):')
print(rateDevEuc[['kValue','errorRate','positiveRate']])
print('Time to run k-NN for 5 iterations: %s' %time_)
print('kNN found the best error rate of %s' % rateDevEuc['errorRate'].min())


print('\nAverage perceprton-Smart implementation (5 epoch):')
start = time.time()
w, devErrList = perc_avg_smart(oneHotTrain, oneHotDev)
time_ = time.time()-start
print('Time to run perceptron for 5 iterations: %s' %time_)
print('Perceptron found the best error rate of %s' %(devErrList.min()))
print('Perceprton weight vecor:')
print(w[devErrList.argmin()])