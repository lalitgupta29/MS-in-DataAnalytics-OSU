import numpy as np
import time

# define function to get mapping
def getMapping(trainingData, fields):
  data_list = [list(tup) for tup in trainingData[fields]]

  mapping = {}

  for row in data_list:
    for i, j in enumerate(row):
      feature = (i, j)
      if feature not in mapping:
        mapping[feature] = len(mapping) # insert a new feature in the index
  return mapping

# define one-hot encoder
def oneHotEncoding(dataset, mapping, fields):
    data_list = [list(tup) for tup in dataset[fields]]
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

# function to get predicted outcomes based on Manhattan distance in k nearest neighbour algorithm
def manhDist(trainSet, testSet, k):
    predictTest=np.zeros(len(testSet))   
    for indx, row in enumerate(testSet):
        manDist = np.sum(abs(trainSet[:,:-1]-row), axis=1)
        if k < len(manDist):
            topKIndx = np.argpartition(manDist, k)[:k]
        else:
            topKIndx = np.argpartition(manDist, len(manDist)-1)
        if trainSet[:,-1][topKIndx].mean() > 0.5:
            predictTest[indx] = 1
    return predictTest

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

# Data pre-processing

# Question 1.1: Positive % of training data and dev data
print('\nQuestion 1.1:')
print('No. of positive % (cases with income >50K) for training data: '
      + str(len(train[train['income'] == ' >50K'])*100.0/len(train)) + ' %')

print('No. of positive % (cases with income >50K) for dev data: '
      + str(len(dev[dev['income'] == ' >50K'])*100.0/len(dev)) + ' %\n')

# Question 1.2: youngest and oldest ages in training set. Least and most
# amount of work per week people do
print('Question 1.2:')
print('The youngest and oldest ages in training set are: '+str(train['age'].min())
      +' and ' +str(train['age'].max()))

print('The least and most amount of work per week in training set are: '
      +str(train['workHours'].min()) +' and ' +str(train['workHours'].max()))


# Question 2.3: Evaluate k-NN on the dev set and report the error rate 
# and predicted positive rate for k = 1, 3, 5, 7, 9, 99, 999, 9999

# get feature map 
fields = ['sector', 'education', 'maritalStatus', 'occupation', 'race', 'gender', 'country']
featureMap = getMapping(train, fields)

# get one-hot encoded train, dev and test sets
oneHotTrain = oneHotEncoding(train, featureMap, fields)
oneHotDev = oneHotEncoding(dev, featureMap, fields)
oneHotTest = oneHotEncoding(test, featureMap, fields)

# convert income to binary format
trainIncome = np.array([0 if x == ' <=50K' else 1 for x in train['income']])
devIncome = np.array([0 if x == ' <=50K' else 1 for x in dev['income']])

# get scaling factor for age and work-hours
sFactorAge = max(train['age'])
sFactorHours = max(train['workHours'])

# get final one-hot format
oneHotTrain = np.column_stack((train['age'].astype('f')/sFactorAge, train['workHours'].astype('f')/sFactorHours, 
                               oneHotTrain, trainIncome.astype('f')))
oneHotDev = np.column_stack((dev['age'].astype('f')/sFactorAge, dev['workHours'].astype('f')/sFactorHours, oneHotDev, 
                             devIncome.astype('f')))
oneHotTest = np.column_stack((test['age'].astype('f')/sFactorAge, test['workHours'].astype('f')/sFactorHours, oneHotTest))

# error and positive rate on dev set: Euclidean distance
kValues = [1,3,5,7,9,99,999,9999]
rateDevEuc = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time()    
    predictDev=eucDist(oneHotTrain, oneHotDev[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDev[:,-1])*100.00/len(oneHotDev)
    positiveRate = sum(predictDev ==1)*100.00/len(oneHotDev)
    rateDevEuc[i] = (k,errorRate,positiveRate,time_)

print('\nQuestion 2.3: k-values, predicted error rate and positive rate on dev set (Euclidean Distance):')
print(rateDevEuc[['kValue','errorRate','positiveRate']])

# error rate for training set
kValues = [1,3,5,7,9,99,999,9999]
rateTrainEuc = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','time'), 'formats':('i4','f','f')})
for i, k in enumerate(kValues):
    start = time.time()
    predictTrain=eucDist(oneHotTrain,oneHotTrain[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictTrain != oneHotTrain[:,-1])*100.00/len(oneHotTrain)
#     positiveRate = sum(predictTrain ==1)*100.00/len(oneHotTrain)
    rateTrainEuc[i] = (k,errorRate,time_)

print('\nQuestion 2.4: Training and dev error rates (Euclidean Distance):')
print(rateTrainEuc[['kValue','errorRate']], rateDevEuc[['kValue','errorRate']])

# error rate and positive rate on Dev set: Manhattan Distance
kValues = [1,3,5,7,9,99,999,9999]
rateDevManh = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time() 
    predictDev=manhDist(oneHotTrain, oneHotDev[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDev[:,-1])*100.00/len(oneHotDev)
    positiveRate = sum(predictDev ==1)*100.00/len(oneHotDev)
    rateDevManh[i] = (k,errorRate,positiveRate,time_)

print('\nQuestion 2.6: k-values, predicted error rate and positive rate on dev set (Manhattan distance):')
print(rateDevManh[['kValue','errorRate','positiveRate']])

# redo the evaluation using all binarized features
fields = ['age', 'sector', 'education', 'maritalStatus', 'occupation', 'race', 
          'gender', 'workHours', 'country']
featureMapAll = getMapping(train, fields)

oneHotTrainAll = oneHotEncoding(train, featureMapAll, fields)
oneHotDevAll = oneHotEncoding(dev, featureMapAll, fields)

oneHotTrainAll = np.column_stack((oneHotTrainAll, trainIncome.astype('f')))
oneHotDevAll = np.column_stack((oneHotDevAll, devIncome.astype('f')))

# error and positive rate on dev set (All binarized features): Euclidean distance
kValues = [1,3,5,7,9,99,999,9999]
# kValues = [1]
rateDevAllEuc = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time()    
    predictDev=eucDist(oneHotTrainAll, oneHotDevAll[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDevAll[:,-1])*100.00/len(oneHotDevAll)
    positiveRate = sum(predictDev ==1)*100.00/len(oneHotDevAll)
    rateDevAllEuc[i] = (k,errorRate,positiveRate,time_)

print('\nQuestion 2.7: k-values, predicted error and positive rate on dev set (All binarized features):')
print(rateDevAllEuc[['kValue','errorRate','positiveRate']])


## Question 3 - Deployment

# error and positive rate on dev set: Euclidean distance
kValues = [1,3,5,7,9,99,159,179,199,259,599,999,9999]
rateDevEuc = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time()    
    predictDev=eucDist(oneHotTrain, oneHotDev[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDev[:,-1])*100.00/len(oneHotDev)
    positiveRate = sum(predictDev ==1)*100.00/len(oneHotDev)
    rateDevEuc[i] = (k,errorRate,positiveRate,time_)

print('\nQuestion 3: \nk values and corresponding error rate and positive ratio with Euclidean Distance on Dev dataset:')
print(rateDevEuc[['kValue','errorRate','positiveRate']])

# error and positive rate on dev set: Manhattan distance
kValues = [1,3,5,7,9,99,159,179,199,259,599,999,9999]
rateDevManh = np.zeros(len(kValues), dtype={'names':('kValue','errorRate','positiveRate','time'), 
                                        'formats':('i4','f','f','f')})
for i, k in enumerate(kValues):
    start = time.time() 
    predictDev=manhDist(oneHotTrain, oneHotDev[:,:-1],k)
    time_ = time.time()-start    
    
    errorRate = sum(predictDev != oneHotDev[:,-1])*100.00/len(oneHotDev)
    positiveRate = sum(predictDev == 1)*100.00/len(oneHotDev)
    rateDevManh[i] = (k,errorRate,positiveRate,time_)

print('\nk values and corresponding error rate and positive ratio with Manhattan Distance on Dev dataset:')
print(rateDevManh[['kValue','errorRate','positiveRate']])

# k-NN on test data using manhattan distance and k=179
kValues = [179]
rateTestManh = np.zeros(len(kValues), dtype={'names':('kValue','positiveRate','time'), 
                                        'formats':('i4','f','f')})
for i, k in enumerate(kValues):
    start = time.time() 
    predictTest=manhDist(oneHotTrain, oneHotTest,k)
    time_ = time.time()-start    
    
    positiveRate = sum(predictTest == 1)*100.00/len(oneHotTest)
    rateTestManh[i] = (k,positiveRate,time_)

print('\nPositive ratio on test:')
print(positiveRate)

# output the file
income = map(lambda x: ' <=50K' if x == 0 else ' >50K', predictTest)
testPredicted = np.column_stack((np.array(test.tolist()),income))
np.savetxt('income.test.predicted',testPredicted,fmt='%s',delimiter=',')


## Question 4.4 How many seconds does it take to print the training and dev errors for k = 99 on ENGR servers? 
start = time.time()
predictDev=eucDist(oneHotTrain, oneHotDev[:,:-1],99)
errorRate = sum(predictDev != oneHotDev[:,-1])*100.00/len(oneHotDev)
time_ = time.time() - start
print('\nQuestion 4.4: \nIt took '+str(time_)+' seconds to get dev error for k = 99.')

start = time.time()
predictTrain=eucDist(oneHotTrain,oneHotTrain[:,:-1],k)
errorRate = sum(predictTrain != oneHotTrain[:,-1])*100.00/len(oneHotTrain)
time_ = time.time()-start    
print('It took '+str(time_)+' seconds to get training error for k = 99.')

