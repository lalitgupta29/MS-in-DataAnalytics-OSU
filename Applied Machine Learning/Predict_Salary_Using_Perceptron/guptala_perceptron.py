import numpy as np

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

# define basic perceptron
def perc_vanilla(oneHotTrain, oneHotDev, epochMax = 5, w0=0, printErr=True):
    wFinal = np.zeros((epochMax, len(oneHotTrain[0])-1))
    w = np.zeros(len(oneHotTrain[0])-1)  # [0,0]
    w[0] = w0
    devErrList = []
    epoch, total, updates = 0, 0, 1
    while ((updates > 0) & (epoch < epochMax)):
        epoch += 1
        updates = 0
        for i, x in enumerate(oneHotTrain[:,:-1]):
            y = oneHotTrain[i][-1]
            if x.dot(w)*y <= 0:
                w += y*x
                updates += 1

        # get dev error rate
        devPredict = get_predict(oneHotDev[:,:-1], w)
        devErrorRate, devPositive = get_error_rate(devPredict, oneHotDev)
        devErrList.append(devErrorRate)
        wFinal[epoch-1] = w
        
        # print updates and error rates
        if printErr:
            print "epoch %d, updates %d (%s%%), dev error %s%% (+:%s%%)" % (epoch, updates, 
                                                                    str(updates*100.0/len(oneHotTrain)), 
                                                                    str(devErrorRate),str(devPositive))
    return wFinal, np.array(devErrList)

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

# define average perceptron (smart version) with shuffling
def perc_avg_shuffle(oneHotTrain, oneHotDev, epochMax = 5, w0=0):
    np.random.seed(1959)
    devErrList = []
    devPosList = []
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
        np.random.shuffle(oneHotTrain)  # shuffle before each epoch
        # get dev error rate
        devPredict = get_predict(oneHotDev[:,:-1], wFinal[epoch-1])
        devErrorRate, devPositive = get_error_rate(devPredict, oneHotDev)
        devErrList.append(devErrorRate)
        devPosList.append(devPositive)

    return wFinal, np.array(devErrList), np.array(devPosList)

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

test = np.genfromtxt('income.test.blind', delimiter=',',
                      dtype={'names':('age', 'sector', 'education', 'maritalStatus',
                                      'occupation', 'race', 'gender', 'workHours', 'country'),
                             'formats':('i4', 'U30','U30','U30', 'U30','U30', 'U30', 'i4',
                                        'U30')})

# get feature map and reverse map
fields = ['age', 'sector', 'education', 'maritalStatus', 'occupation', 'race', 'gender', 'workHours', 'country']
featureMap, featureReMap = getMapping(train[fields])

# get one-hot encoded train, dev and test sets
oneHotTrain = oneHotEncoding(train[fields], featureMap)
oneHotDev = oneHotEncoding(dev[fields], featureMap)
oneHotTest = oneHotEncoding(test[fields], featureMap)

oneHotTrain = np.column_stack((np.ones(len(oneHotTrain)), oneHotTrain, 
                               np.array([-1 if x == ' <=50K' else 1 for x in train['income']])))
oneHotDev = np.column_stack((np.ones(len(oneHotDev)), oneHotDev,
                            np.array([-1 if x == ' <=50K' else 1 for x in dev['income']])))
oneHotTest = np.column_stack((np.ones(len(oneHotTest)),oneHotTest))

## Question 1.2
print("There are total %s binary features after binarizing all fields (including age and work-hours)" %len(oneHotTrain[0,:-1]))

## Question 2.1
print('\nBasic perceprton (5 epoch):')
perc_vanilla(oneHotTrain, oneHotDev)

## Question 2.2
print('\nAverage perceprton-Smart implementation (5 epoch):')
w, devErrList = perc_avg_smart(oneHotTrain, oneHotDev)

print('\nAverage perceprton-Smart implementation (with bias 3):')
perc_avg_smart(oneHotTrain, oneHotDev, w0=3)

## Question 2.4
# five most positive/negative features
pos = np.argsort(w[-1])[-5:]-1
neg = np.argsort(w[-1])[:6]-1
neg = neg[neg >= 0]

posFeatures = [featureReMap[key] for key in pos]
negFeatures = [featureReMap[key] for key in neg]

print('\nFive most postive features: \n %s' %posFeatures)
print('Five most negative features: \n %s' %negFeatures)

## Question 2.5
print('\nWeight for females and males: (%s, %s)' %tuple(np.take(w[-1],[featureMap[(6,' Female')]+1, featureMap[(6,' Male')]+1])))

## Question 2.6
print('Feature weight for bias dimension (average perceptron-smart implementation) is: %s' %w[-1][0])

## Question 4.1
# reorder training data to all positives and then all negatives
trainSorted = oneHotTrain[(-oneHotTrain[:,-1]).argsort()]

print('\nPerceptron after reordering training data to all positive and then all negatives\nBasic Perceptron:')
perc_vanilla(trainSorted, oneHotDev)

print('\nAverage Perceptron:') 
perc_avg_smart(trainSorted, oneHotDev)

## Question 4.2
print('\nExperimenting with fetures')

print('Adding age and work-hours as new features:')
trainExp = np.column_stack((oneHotTrain[:,:-1],train['age'], train['workHours'],oneHotTrain[:,-1]))
devExp = np.column_stack((oneHotDev[:,:-1],dev['age'], dev['workHours'],oneHotDev[:,-1]))
perc_avg_smart(trainExp, devExp)

print('\nCentering the numerical features:')
trainExp[:,-3:-1] -= trainExp[:,-3:-1].mean(axis=0)
devExp[:,-3:-1] -= devExp[:,-3:-1].mean(axis=0)
perc_avg_smart(trainExp, devExp)

print('\nMaking numerical features unit variance:')
trainExp[:,-3:-1] = np.divide(trainExp[:,-3:-1], trainExp[:,-3:-1].std(axis=0)[np.newaxis])
devExp[:,-3:-1] = np.divide(devExp[:,-3:-1], devExp[:,-3:-1].std(axis=0)[np.newaxis])
trainExp[:,-3:-1][np.isnan(trainExp[:,-3:-1])] = 0
devExp[:,-3:-1][np.isnan(devExp[:,-3:-1])] = 0
perc_avg_smart(trainExp, devExp)

print('\nAdding some binary combination features:')
newTrain1 = ((train['education'] == ' Doctorate') & (train['sector'] == ' Self-emp-inc'))
newDev1 = ((dev['education'] == ' Doctorate') & (dev['sector'] == ' Self-emp-inc'))
newTrain1 = np.array([1 if x else 0 for x in newTrain1])
newDev1 = np.array([1 if x else 0 for x in newDev1])
trainExp = np.column_stack((trainExp[:,:-1], newTrain1, trainExp[:,-1]))
devExp = np.column_stack((devExp[:,:-1], newDev1, devExp[:,-1]))
perc_avg_smart(trainExp, devExp)

# get best predicted model based on shuffling, if not found by shuffling use smart perceptron
print('\nRunning smart perceptron by shuffling training data on each epoch to see if a dev error rate of <13.5 can be found over 100 iterations.')
count = 0
notFound = True
trainShuffle = oneHotTrain
while ((count < 100) & (notFound)):
    count += 1
    w, devErrList, devPositive = perc_avg_shuffle(trainShuffle, oneHotDev, 10)
    if devErrList.min() <= 13.5:
        notFound = False
        wFinal = w[devErrList.argmin()]
        posRate = devPositive[devErrList.argmin()]
        print('Shuffling found an error rate of %s after %s iterations' %(devErrList.min(), count))
        print('Predicted positive rate on dev: %s' %posRate)

if notFound == False:
    predict = get_predict(oneHotTest, wFinal)
    
else:
    w, devErrList = perc_avg_smart(oneHotTrain, oneHotDev,printErr=False)
    wFinal = w[devErrList.argmin()]
    predict = get_predict(oneHotTest, wFinal)

income = map(lambda x: ' >50K' if x == 1 else ' <=50K', predict)
print('Predicted positive rate on test set: %s' %(sum(np.array(predict) == 1)*100.0/len(predict)))
testPredicted = np.column_stack((np.array(test.tolist()),income))
np.savetxt('income.test.predicted',testPredicted,fmt='%s',delimiter=',')
print('\n income.test.predicted file saved successfully.')