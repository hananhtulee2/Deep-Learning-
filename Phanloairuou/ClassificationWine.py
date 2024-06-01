# declare function and libraries that used in the program
# from numpy import mean
# from numpy import std
from numpy import mean , std
from pandas import read_csv
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import preprocessing, pipeline , tree, neighbors, neural_network, ensemble, linear_model

#load dataset from internet
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header = None)
data = df.values
# dataset has 178 data point (samples)
# each sample includes 13 values of input X and one value of Y
# there are three classes for classification
X, y = data[:,:-1] ,data[:,-1]
X= X.astype('float')
y = preprocessing.LabelEncoder().fit_transform(y.astype('str'))

# data transformation and feature engineer
dataTransform = list()
# dataTransform.append(('f1',preprocessing.KBinsDiscretizer()))
# dataTransform.append(('f2',preprocessing.MinMaxScaler()))
# dataTransform.append(('f3',preprocessing.MaxAbsScaler()))
# dataTransform.append(('f4',preprocessing.PolynomialFeatures()))
dataTransform.append(('f5',preprocessing.QuantileTransformer(n_quantiles = 142)))
#dataTransform.append(('f6',preprocessing.StandardScaler()))
# create the feature union
fu = pipeline.FeatureUnion(dataTransform)

# create and define the machine learning model for classification
# model =tree.DecisionTreeClassifier()
model =tree.ExtraTreeClassifier()
# model =neighbors.KNeighborsClassifier()
# model =linear_model.PassiveAggressiveClassifier()
# model = neural_network.MLPClassifier()
# model = ensemble.BaggingClassifier()
# model = ensemble.ExtraTreesClassifier()
# model = ensemble.RandomForestClassifier()
#define the pipline
thanhphan =list()
thanhphan.append(('feature', fu ))
thanhphan.append(('model',model))
tienTrinh =pipeline.Pipeline(steps= thanhphan)

#define the cross-validation procedure
protocol = RepeatedStratifiedKFold(n_splits=5,n_repeats=10)

#evaluate the performance of trained classification model
scores = cross_val_score(tienTrinh, X, y, scoring= 'accuracy', cv= protocol)

#print the numerical results
print('Average accuracy : %0.1f' %(mean(scores)*100))
print('Average accuracy : %0.1f' %(std(scores)*100))