import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
import statsmodels.formula.api as sm

# define function to get mapping
def getMapping(trainingData, mapping = {}, reMap = {}):
    data_list = [list(tup[1]) for tup in trainingData.iterrows()]

    for row in data_list:
        for i, j in enumerate(row):
            feature = (i, j)
            if feature not in mapping:
                reMap[len(mapping)] = feature
                mapping[feature] = len(mapping) # insert a new feature in the index
    return mapping, reMap

# define one-hot encoder
def oneHotEncoding(dataset, mapping):
    data_list = [list(x[1]) for x in dataset.iterrows()]
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
    return pd.DataFrame(binaryData)

# define function to calculate Root Mean Square Error
def rmse(predict, actual):
    return np.sqrt(np.mean((predict-actual)**2))

# Load data
train = pd.read_csv('my_train.csv')
dev =  pd.read_csv('my_dev.csv')
test = pd.read_csv('test.csv')

# update nan to 0
train.fillna(0, inplace=True); dev.fillna(0, inplace=True); test.fillna(0, inplace=True)

# convert to log scale
train.iloc[:,-1] = train.iloc[:,-1].apply(np.log)
dev.iloc[:,-1] = dev.iloc[:,-1].apply(np.log)

# get mapping and reverse mapping 
featureMap, featureReMap = getMapping(train.iloc[:,1:-1],{},{})

print('There are total %s features' %len(featureMap))

# get number of features per field
print('\nNo. of features for each field:')
featureField = []
for i in range(0,79):
    count = 0
    for key in featureMap.iterkeys():
        if key[0] == i:
            count += 1
    print("Variable %s has %s features" %(train.columns[i+1], count))
    featureField.append([i+2, count])

# get one-hot encoded train, dev and test sets
OneHotTrainNaive = oneHotEncoding(train.iloc[:,1:-1], featureMap)
OneHotDevNaive = oneHotEncoding(dev.iloc[:,1:-1], featureMap)
OneHotTestNaive = oneHotEncoding(test.iloc[:,1:], featureMap)

OneHotTrainNaive = pd.concat([OneHotTrainNaive, train.iloc[:,-1]], axis=1)
OneHotDevNaive = pd.concat([OneHotDevNaive, dev.iloc[:,-1]], axis=1)
OneHotTestNaive = pd.concat([OneHotTestNaive], axis=1)

# fitting regression model with sklearn
linearReg = LR()
linearReg.fit(OneHotTrainNaive.iloc[:,:-1],OneHotTrainNaive.iloc[:,-1])
devPred = linearReg.predict(OneHotDevNaive.iloc[:,:-1])
rmsleVan = rmse(devPred,OneHotDevNaive.iloc[:,-1])
print('\nRoom Mean Square Log Error for Naive implementations: %s' %(rmsleVan))

# get top 10 positive and negative features
coeff = linearReg.coef_
topFeat = np.argsort(coeff)[-10:]
bottomFeat = np.argsort(coeff)[:10]

print("\nTop 10 Positive Features:")
for x in topFeat:
    print("Variable: %s, Value: %s" %(train.columns[featureReMap[x][0]+1], featureReMap[x][1]))

print("\nTop 10 Negative Features:")
for x in bottomFeat:
    print("Variable: %s, Value: %s" %(train.columns[featureReMap[x][0]+1], featureReMap[x][1]))    

# feature weight for bias 
print("\nFeature weight for bias dimension: %s" %linearReg.intercept_)

# plot Sale Price
## uncomment below section on code to plot histogram of Sale Price
# print("\nLet's look at the histogram for Sale Price:")
# n, bins, patches = plt.hist(train.SalePrice, 20, facecolor='g', alpha=0.75)
# plt.title('Sale Price')
# plt.axis([10, 14, 0, 250])
# plt.show()

# make prediction and save prediction file
testPred = np.exp(linearReg.predict(OneHotTestNaive))
test_predict = pd.concat([test.iloc[:,0],pd.DataFrame(testPred, columns=['SalePrice'])], axis = 1)
test_predict.to_csv('test_predicted.csv', index = False)
print('\nPrediction file "test_predicted.csv" saved.')

## Smarter binarization

# Load data
# train = pd.read_csv('my_train.csv')
# dev =  pd.read_csv('my_dev.csv')
# test = pd.read_csv('test.csv')

print('\nSmarter binarization:')
# convert year features to age
colsForAge = ["YearBuilt", "GarageYrBlt", "YrSold", "YearRemodAdd"]
train[colsForAge] = train[colsForAge].apply(lambda x: pd.datetime.now().year - x)
dev[colsForAge] = dev[colsForAge].apply(lambda x: pd.datetime.now().year - x)
test[colsForAge] = test[colsForAge].apply(lambda x: pd.datetime.now().year - x)

# get numerical and categorical features
featuresNum = ["LotFrontage","LotArea","YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF",
               "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
               "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", 
               "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", 
               "MiscVal", "YrSold"]

featuresCat = []
for feature in train.columns[1:-1]:
    if feature not in featuresNum:
        featuresCat.append(feature)

# rearrange columns
id_ = train.Id
price = train.SalePrice
train = train[np.concatenate((train[featuresCat].columns, train[featuresNum].columns))]
train = pd.concat([id_,train,price], axis = 1)

id_ = dev.Id
price = dev.SalePrice
dev = dev[np.concatenate((dev[featuresCat].columns, dev[featuresNum].columns))]
dev = pd.concat([id_,dev,price], axis = 1)

id_ = test.Id
test = test[np.concatenate((test[featuresCat].columns, test[featuresNum].columns))]
test = pd.concat([id_,test], axis = 1)

# get mapping and reverse mapping 
featureMap = {}; featureReMap = {}
featureMap, featureReMap = getMapping(train[featuresCat], {}, {})

# get one-hot encoded train, dev and test sets
oneHotTrainSmart = oneHotEncoding(train[featuresCat], featureMap)
oneHotDevSmart = oneHotEncoding(dev[featuresCat], featureMap)
oneHotTestSmart = oneHotEncoding(test[featuresCat], featureMap)

oneHotTrainSmart = pd.concat([oneHotTrainSmart, train[featuresNum], train.iloc[:,-1]], axis=1)
oneHotDevSmart = pd.concat([oneHotDevSmart, dev[featuresNum], dev.iloc[:,-1]], axis=1)
oneHotTestSmart = pd.concat([oneHotTestSmart, test[featuresNum]], axis=1)

print('Total features: %s' %(len(oneHotTrainSmart.columns) -1))

# implement linear regression
linearRegSmart = LR()
linearRegSmart.fit(oneHotTrainSmart.iloc[:,:-1],oneHotTrainSmart.iloc[:,-1])
devPred = linearRegSmart.predict(oneHotDevSmart.iloc[:,:-1])

# get RMSLE
rmsle = rmse(devPred,oneHotDevSmart.iloc[:,-1])
print('\nRoom Mean Square Log Error for Naive implementations: %s' %(rmsle))

# top 10 positive and negative features
coeff = linearRegSmart.coef_
topFeat = np.argsort(coeff)[-10:]
bottomFeat = np.argsort(coeff)[:10]

print("\nTop 10 Positive Features:")
for x in topFeat:
    if x < len(featureMap):
        print("Variable: %s, Value: %s" %(train.columns[featureReMap[x][0]+1], featureReMap[x][1]))
    else:
        print("Variable: %s" %(oneHotTrainSmart.columns[x]))

print("\nTop 10 Negative Features:")
for x in bottomFeat:
    if x < len(featureMap):
        print("Variable: %s, Value: %s" %(train.columns[featureReMap[x][0]+1], featureReMap[x][1]))
    else:
        print("Variable: %s" %(oneHotTrainSmart.columns[x]))

# feature weight for bias 
print("\nFeature weight for bias dimension: %s" %linearRegSmart.intercept_)

# get prediction on test
testPred = np.exp(linearRegSmart.predict(oneHotTestSmart))
test_predict = pd.concat([test.iloc[:,0],pd.DataFrame(testPred, columns=['SalePrice'])], axis = 1)
test_predict.to_csv('test_smart_predicted.csv', index = False)
print('\nPrediction file "test_smart_predicted.csv" saved.')

## Experimentation
from sklearn.linear_model import Ridge
alpha_set = [0.25, 0.5, 1, 2, 5, 10]
print('\nNaive Implementatio:')
for alpha_ in alpha_set:
    regLR = Ridge(alpha=alpha_)
    regLR.fit(OneHotTrainNaive.iloc[:,:-1],OneHotTrainNaive.iloc[:,-1])
    devPred = regLR.predict(OneHotDevNaive.iloc[:,:-1])
    rmsle = rmse(devPred,OneHotDevNaive.iloc[:,-1])
    print('Room Mean Square Log Error for regularized implementation (with alpha = %s): %s' %(alpha_, rmsle))

print('\nSmart Implementation')
alpha_set = [0.25, 0.5, 1, 2, 5, 10]
for alpha_ in alpha_set:
    regLR = Ridge(alpha=alpha_)
    regLR.fit(oneHotTrainSmart.iloc[:,:-1],oneHotTrainSmart.iloc[:,-1])
    devPred = regLR.predict(oneHotDevSmart.iloc[:,:-1])
    rmsle = rmse(devPred,oneHotDevSmart.iloc[:,-1])
    print('Room Mean Square Log Error for regularized implementation (with alpha = %s): %s' %(alpha_, rmsle))


# non linear features

# Load data
train = pd.read_csv('my_train.csv')
dev =  pd.read_csv('my_dev.csv')
test = pd.read_csv('test.csv')

# update nan to 0
train.fillna(0, inplace=True); dev.fillna(0, inplace=True); test.fillna(0, inplace=True)

# get total square feet
train.insert(loc = len(train.iloc[0])-1, column = 'TotalSF', 
             value = train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF'])
dev.insert(loc = len(dev.iloc[0])-1, column = 'TotalSF', 
             value = dev['TotalBsmtSF']+dev['1stFlrSF']+dev['2ndFlrSF'])
test.insert(loc = len(test.iloc[0])-1, column = 'TotalSF', 
             value = test['TotalBsmtSF']+test['1stFlrSF']+test['2ndFlrSF'])

# drop columns
train = train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)
dev = dev.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)
test = test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)

# get numerical and categorical features
featuresNum = ["LotFrontage","LotArea","YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF",
               "TotalSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
               "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", 
               "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", 
               "MiscVal", "YrSold"]
featuresCat = []
for feature in train.columns[1:-1]:
    if feature not in featuresNum:
        featuresCat.append(feature)

# convert "LotArea" to log scale
train.LotArea = np.log(train.LotArea)
dev.LotArea = np.log(dev.LotArea)
test.LotArea = np.log(test.LotArea)

# convert "TotalSF"
train.TotalSF = np.sqrt(train.TotalSF)
dev.TotalSF = np.sqrt(dev.TotalSF)
test.TotalSF = np.sqrt(test.TotalSF)

# convert Sale Price to log scale
train.iloc[:,-1] = train.iloc[:,-1].apply(np.log)
dev.iloc[:,-1] = dev.iloc[:,-1].apply(np.log)

# get mapping and reverse mapping
featureMap = {}; featureReMap = {}
featureMap, featureReMap = getMapping(train[featuresCat], {}, {})

# get one-hot encoded train, dev and test sets
oneHotTrain = oneHotEncoding(train[featuresCat], featureMap)
oneHotDev = oneHotEncoding(dev[featuresCat], featureMap)
oneHotTest = oneHotEncoding(test[featuresCat], featureMap)

oneHotTrain = pd.concat([oneHotTrain, train[featuresNum], train.iloc[:,-1]], axis=1)
oneHotDev = pd.concat([oneHotDev, dev[featuresNum], dev.iloc[:,-1]], axis=1)
oneHotTest = pd.concat([oneHotTest, test[featuresNum]], axis=1)

alpha_ = 0.5
regLR = Ridge(alpha=alpha_)
regLR.fit(oneHotTrain.iloc[:,:-1],oneHotTrain.iloc[:,-1])
devPred = regLR.predict(oneHotDev.iloc[:,:-1])
rmsle = rmse(devPred,oneHotDev.iloc[:,-1])
print('\nRMSLE after scaling lot area and square footage (with alpha = %s): %s' %(alpha_, rmsle))

# adding new column remod
oneHotTrain.insert(loc = len(oneHotTrain.iloc[0])-1, column = 'remod', 
             value = oneHotTrain['YearRemodAdd']-oneHotTrain['YearBuilt'])
oneHotDev.insert(loc = len(oneHotDev.iloc[0])-1, column = 'remod', 
             value = oneHotDev['YearRemodAdd']-oneHotDev['YearBuilt'])

alpha_ = 0.5
regLR = Ridge(alpha=alpha_)
regLR.fit(oneHotTrain.iloc[:,:-1],oneHotTrain.iloc[:,-1])
devPred = regLR.predict(oneHotDev.iloc[:,:-1])
rmsle = rmse(devPred,oneHotDev.iloc[:,-1])
print('RMSLE after adding year since remodeled (with alpha = %s): %s' %(alpha_, rmsle))

## further improvement
# Load data
train = pd.read_csv('my_train.csv')
dev =  pd.read_csv('my_dev.csv')
test = pd.read_csv('test.csv')

# update nan to 0
train.fillna(0, inplace=True); dev.fillna(0, inplace=True); test.fillna(0, inplace=True)

# get total square feet
train.insert(loc = len(train.iloc[0])-1, column = 'TotalSF', 
             value = train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF'])
dev.insert(loc = len(dev.iloc[0])-1, column = 'TotalSF', 
             value = dev['TotalBsmtSF']+dev['1stFlrSF']+dev['2ndFlrSF'])
test.insert(loc = len(test.iloc[0])-1, column = 'TotalSF', 
             value = test['TotalBsmtSF']+test['1stFlrSF']+test['2ndFlrSF'])

# drop columns
train = train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)
dev = dev.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)
test = test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1)

featuresNum = ["LotFrontage","LotArea","YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF",
               "TotalSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
               "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", 
               "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", 
               "MiscVal", "YrSold"]
featuresCat = []
for feature in train.columns[1:-1]:
    if feature not in featuresNum:
        featuresCat.append(feature)

train.GrLivArea[train.GrLivArea > 5000]
train = train.drop(train.loc[train['MSZoning']=='C (all)'].index)
train = train.drop(train.loc[train['GrLivArea']>5000].index)
train = train.drop(train.loc[train['TotRmsAbvGrd']>13].index)
train = train.drop(train.loc[train['MasVnrArea']>1500].index)
train = train.drop(train.loc[train['BsmtFinSF1']>1500].index)
train = train.drop(train.loc[train['LotFrontage']>300].index)
train = train.drop(train.loc[train['EnclosedPorch']>500].index)
train = train.drop(train.loc[train['BsmtFinSF2']>1400].index)
train = train.drop(train.loc[train['TotalSF']>7000].index)
train = train.drop(train.loc[train['GarageCars']>=4].index)
train = train.drop(train.loc[train['GarageArea']>1300].index)
train = train.drop(train.loc[train['FullBath']==0].index)
train = train.drop(train.loc[train['BsmtFullBath']>=2].index)
train = train.reset_index(drop=True)

# convert "LotArea" to log scale
train.LotArea = np.log(train.LotArea)
dev.LotArea = np.log(dev.LotArea)
test.LotArea = np.log(test.LotArea)

# convert "TotalSF" to log scale
train.TotalSF = np.sqrt(train.TotalSF)
dev.TotalSF = np.sqrt(dev.TotalSF)
test.TotalSF = np.sqrt(test.TotalSF)

# convert "GrLivArea" to log scale
train.GrLivArea = np.log(train.GrLivArea)
dev.GrLivArea = np.log(dev.GrLivArea)
test.GrLivArea = np.log(test.GrLivArea)

# convert "YearBuilt" to log scale
train.YearBuilt = np.square(train.YearBuilt)
dev.YearBuilt = np.square(dev.YearBuilt)
test.YearBuilt = np.square(test.YearBuilt)

# convert "TotRmsAbvGrd" to log scale
train.TotRmsAbvGrd = np.sqrt(train.TotRmsAbvGrd)
dev.TotRmsAbvGrd = np.sqrt(dev.TotRmsAbvGrd)
test.TotRmsAbvGrd = np.sqrt(test.TotRmsAbvGrd)

# convert "Fireplaces" to log scale
train.Fireplaces = np.sqrt(train.Fireplaces)
dev.Fireplaces = np.sqrt(dev.Fireplaces)
test.Fireplaces = np.sqrt(test.Fireplaces)

# convert "MasVnrArea" to log scale
train.MasVnrArea = np.sqrt(train.MasVnrArea)
dev.MasVnrArea = np.sqrt(dev.MasVnrArea)
test.MasVnrArea = np.sqrt(test.MasVnrArea)

# convert "BsmtFinSF1" to log scale
train.BsmtFinSF1 = np.square(train.BsmtFinSF1)
dev.BsmtFinSF1 = np.square(dev.BsmtFinSF1)
test.BsmtFinSF1 = np.square(test.BsmtFinSF1)

# convert "OpenPorchSF" to log scale
train.OpenPorchSF = np.sqrt(train.OpenPorchSF)
dev.OpenPorchSF = np.sqrt(dev.OpenPorchSF)
test.OpenPorchSF = np.sqrt(test.OpenPorchSF)

# convert "WoodDeckSF" to log scale
train.WoodDeckSF = np.sqrt(train.WoodDeckSF)
dev.WoodDeckSF = np.sqrt(dev.WoodDeckSF)
test.WoodDeckSF = np.sqrt(test.WoodDeckSF)

# convert "BsmtUnfSF" to log scale
train.BsmtUnfSF = np.square(train.BsmtUnfSF)
dev.BsmtUnfSF = np.square(dev.BsmtUnfSF)
test.BsmtUnfSF = np.square(test.BsmtUnfSF)

# convert "LotFrontage" to log scale
train.LotFrontage = np.square(train.LotFrontage)
dev.LotFrontage = np.square(dev.LotFrontage)
test.LotFrontage = np.square(test.LotFrontage)

# convert "EnclosedPorch" to log scale
train.EnclosedPorch = np.sqrt(train.EnclosedPorch)
dev.EnclosedPorch = np.sqrt(dev.EnclosedPorch)
test.EnclosedPorch = np.sqrt(test.EnclosedPorch)

# drop columns
train = train.drop(['LowQualFinSF','BsmtHalfBath','BsmtFinSF2','MiscVal'], axis = 1)
dev = dev.drop(['LowQualFinSF','BsmtHalfBath','BsmtFinSF2','MiscVal'], axis = 1)
test = test.drop(['LowQualFinSF','BsmtHalfBath','BsmtFinSF2','MiscVal'], axis = 1)

# convert sale price to log scale
train.iloc[:,-1] = train.iloc[:,-1].apply(np.log)
dev.iloc[:,-1] = dev.iloc[:,-1].apply(np.log)

for elem in ['LowQualFinSF','BsmtHalfBath','BsmtFinSF2','MiscVal']:
    featuresNum.remove(elem)

# get mapping and reverse mapping
featureMap = {}; featureReMap = {}
featureMap, featureReMap = getMapping(train[featuresCat], {}, {})
len(featureMap), len(featureReMap)

# get one-hot encoded train, dev and test sets
oneHotTrain = oneHotEncoding(train[featuresCat], featureMap)
oneHotDev = oneHotEncoding(dev[featuresCat], featureMap)
oneHotTest = oneHotEncoding(test[featuresCat], featureMap)

oneHotTrain = pd.concat([oneHotTrain, train[featuresNum], train.iloc[:,-1]], axis=1)
oneHotDev = pd.concat([oneHotDev, dev[featuresNum], dev.iloc[:,-1]], axis=1)
oneHotTest = pd.concat([oneHotTest, test[featuresNum]], axis=1)

# reduce features by using statsmodel
oneHotTrainSM = pd.concat([pd.DataFrame(np.ones(len(oneHotTrain)), columns=['bias']), oneHotTrain], axis=1)
oneHotDevSM = pd.concat([pd.DataFrame(np.ones(len(oneHotDev)), columns=['bias']), oneHotDev], axis=1)
oneHotTestSM = pd.concat([pd.DataFrame(np.ones(len(oneHotTest)), columns=['bias']), oneHotTest], axis=1)

# initialize variables
pVal = 0.15
count = 0

# reduce features using bic
while count < len(oneHotTrainSM.columns)-1:
    count += 1
    regOLS = sm.OLS(endog=oneHotTrainSM.iloc[:,-1], exog=oneHotTrainSM.iloc[:,:-1]).fit()
    if regOLS.pvalues.max() > pVal:
        oneHotTrainSM = oneHotTrainSM.drop([regOLS.pvalues.idxmax()], axis=1)
        oneHotDevSM = oneHotDevSM.drop([regOLS.pvalues.idxmax()], axis=1)
        oneHotTestSM = oneHotTestSM.drop([regOLS.pvalues.idxmax()], axis=1)
    else:
        break

if 'bias' in oneHotTrainSM.columns.values:
    oneHotTrainSM = oneHotTrainSM.drop(['bias'], axis=1)
    oneHotDevSM = oneHotDevSM.drop(['bias'], axis=1)
    oneHotTestSM = oneHotTestSM.drop(['bias'], axis=1)

# reassign training, dev and test data
oneHotTrain = oneHotTrainSM
oneHotDev = oneHotDevSM
oneHotTest = oneHotTestSM

regLR = Ridge(alpha=0.5)
regLR.fit(oneHotTrain.iloc[:,:-1],oneHotTrain.iloc[:,-1])
devPred = regLR.predict(oneHotDev.iloc[:,:-1])
rmsle = rmse(devPred,oneHotDev.iloc[:,-1])
print('\nRMSLE of final model on dev set is: %s' %rmsle)

# make final prediction
testPred = np.exp(regLR.predict(oneHotTest))
test_predict = pd.concat([test.iloc[:,0],pd.DataFrame(testPred, columns=['SalePrice'])], axis = 1)
test_predict.to_csv('test_predicted_final.csv', index = False)
print('\nFinal predection file "test_predicted_final.csv" saved.')


