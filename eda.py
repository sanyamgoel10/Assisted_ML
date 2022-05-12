import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
from os.path import exists

# Creating Correlation Grah

def createCorr(df):
    corr = df.corr()
    plt.figure().clear()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

    plt.savefig("static\images\Correlation_Graph.png", dpi=50)


# K-Means Clustering
def kMeansUnsupervised(x, n, key, dary):
    if key == '':
        if n == 0:
            wcss = []
            x = x.values
            i = 1
            while 1 < 2:
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(x)
                wcss.append(kmeans.inertia_)
                if kmeans.inertia_ < 3000:
                    break
                i += 1
            plt.figure().clear()
            plt.plot(range(1, i + 1), wcss)
            plt.title('Within Cluster Sum Of Square Error Graph')
            plt.xlabel('# clusters')
            plt.ylabel('wcss')
            plt.savefig("static\images\kMeansClusteringWCSSGraph.png", dpi=50)
            # plt.show()

            anarray = [x.tostring(), str(x.shape)]
            return anarray
        else:
            x2 = x[0]
            arshape = x[1]
            arshape = eval(arshape)
            x1 = np.fromstring(x2)
            x1 = np.fromstring(x1, dtype=float)
            x1 = np.fromstring(x1, dtype=float).reshape(arshape[0], arshape[1])

            kmeansUser = KMeans(n_clusters=n, init='k-means++', random_state=42)
            kmeansUser.fit(x1)
            y_kmeans = kmeansUser.fit_predict(x1)
            with open("kmeansModelNew.pickle", "wb") as f:
                pickle.dump(kmeansUser, f)
            wcssError = kmeansUser.inertia_
            anarray = [str(y_kmeans), str(wcssError), n]
            return anarray
    else:
        predictionKey = eval(key)
        if exists('kmeansencoder.pickle'):
            with open('kmeansencoder.pickle', 'rb') as f:
                ct = pickle.load(f)
        cat = []
        num = []
        for i in range(len(dary)):
            if dary[i] in [1, 4]:
                cat.append(i)
            elif dary[i] in [2, 3]:
                num.append(i)

        catArray = []
        for i in cat:
            catArray.append(predictionKey[i])
        numArray = []
        for i in num:
            numArray.append(predictionKey[i])
        if cat:
            catArrayNumpy = np.array([catArray])
            catArrayNumpy = ct.transform(catArrayNumpy)

        if num:
            numArrayNumpy = np.array([numArray])

        if cat != [] and num != []:
            t1 = pd.concat([pd.DataFrame(catArrayNumpy), pd.DataFrame(numArrayNumpy)], axis=1, join='inner')
        elif cat != [] and num == []:
            t1 = pd.DataFrame(catArrayNumpy)
        elif cat == [] and num != []:
            t1 = pd.DataFrame(numArrayNumpy)

        predictionDataset = t1.values
        with open("kmeansModelNew.pickle", "rb") as f:
            yKmeans = pickle.load(f)
        y_kmeans1 = yKmeans.predict(predictionDataset)
        return [str(y_kmeans1[0])]


########################################################

# Regression
def linearRegression(x, y, key, dary):
    if not key:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        with open('lnrRegModel.pickle', 'wb') as f:
            pickle.dump(regressor, f)
        y_pred = regressor.predict(x_test)
        np.set_printoptions(precision=2)

        return [mean_squared_error(y_test, y_pred, squared=False), regressor.score(x_test, y_test)]
    else:
        predictionKey = eval(key)
        if exists('supencoder.pickle'):
            with open('supencoder.pickle', 'rb') as f:
                ct = pickle.load(f)
        if exists('supstandardizer.pickle'):
            with open('supstandardizer.pickle', 'rb') as f:
                sc = pickle.load(f)
        cat = []
        num = []
        for i in range(len(dary)):
            if dary[i] in [1, 4]:
                cat.append(i)
            elif dary[i] in [2, 3]:
                num.append(i)

        catArray = []
        for i in cat:
            catArray.append(predictionKey[i])
        numArray = []
        for i in num:
            numArray.append(predictionKey[i])
        catArrayNumpy = np.array([catArray])
        numArrayNumpy = np.array([numArray])
        if exists('supencoder.pickle'):
            catArrayNumpy = ct.transform(catArrayNumpy)
        if exists('supstandardizer.pickle'):
            numArrayNumpy = sc.transform(numArrayNumpy)
        t1 = pd.concat([pd.DataFrame(catArrayNumpy), pd.DataFrame(numArrayNumpy)], axis=1, join='inner')
        with open('lnrRegModel.pickle', 'rb') as f:
            regressor = pickle.load(f)
        results1234 = regressor.predict(t1.values)[0][0]
        return [results1234, key]


########################################################

# SVM Classification
def SVMclassification(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return [str(cm), accuracy_score(y_test, y_pred)]


# Decision Tree Classification
def decisionTreeClassification(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return [str(cm), accuracy_score(y_test, y_pred)]


#################################################################################################################
#################################################################################################################


# Processing

def finddtype(dataset):
    arr = [0] * len(dataset.columns)
    for i in range(len(dataset.columns)):
        # 1 ->  int64 categorical
        # 2 ->  int64 numerical
        # 3 ->  float64 (numerical)
        # 4 ->  string(object) with <2 gap (categorical)
        # 5 ->  string(object) with >=2gap (remove the column)
        if str(dataset[dataset.columns[i]].dtypes) == 'int64':
            unq = dataset[dataset.columns[i]].unique()
            if len(dataset.index) >= 500:
                if len(unq) <= (len(dataset.index) ** (1 / 3)):
                    arr[i] = 1  # int categorical
                else:
                    arr[i] = 2  # int numerical
            else:
                if len(unq) <= (len(dataset.index) ** (2 / 3)):
                    arr[i] = 1  # int categorical
                else:
                    arr[i] = 2  # int numerical

        elif str(dataset[dataset.columns[i]].dtypes) == 'float64':
            arr[i] = 3  # float64 numerical
        elif str(dataset[dataset.columns[i]].dtypes) == 'object':
            df2 = dataset[dataset.columns[i]].sample(frac=0.3)
            false = 0
            for j in df2:
                if j.count(' ') >= 2:
                    false += 1
            if false == 0:
                arr[i] = 4  # string <2gap categorical
            else:
                arr[i] = 5  # string >2gap remove the column
    return arr


def func(yColumnNumber, name, indexColumn):
    dataset = pd.read_csv(name)

    if indexColumn != -1:
        dataset = dataset.drop(dataset.columns[[indexColumn]], axis=1)

    createCorr(dataset)

    # unsupervised learning
    if yColumnNumber == -1:

        allColumnsNamesInitially = dataset.columns.values.tolist()
        countOfDatapoints = len(dataset.index)
        # getting array whose indexes contains values (
        #         # 1 ->  int64 categorical
        #         # 2 ->  int64 numerical
        #         # 3 ->  float64 (numerical)
        #         # 4 ->  string(object) with <2 gap (categorical)
        #         # 5 ->  string(object) with >=2gap (remove the column)
        #         )
        dtypeArray = finddtype(dataset)  # dtypeArray=[2,2,2,1,2]
        dictabc = {1: 'int64 dtype --> Numerical',
                   2: 'int64 dtype --> Categorical',
                   3: 'float64 dtype --> Numerical',
                   4: 'string type --> Categorical',
                   5: 'Long String type --> Removed'}
        newDtypeArray = []
        for i in dtypeArray:
            newDtypeArray.append(dictabc[i])
        dict_from_list = dict(zip(allColumnsNamesInitially, newDtypeArray))
        x = dataset

        # cat is array that contains column number of categorical columns in the dataset
        # num is array that contains column number of numerical columns in the dataset
        cat = []  # [3]
        num = []  # [0,1,2,4]
        for i in range(len(dtypeArray)):
            if dtypeArray[i] == 5:
                pass
            elif dtypeArray[i] in [1, 4]:
                cat.append(i)
            elif dtypeArray[i] in [2, 3]:
                num.append(i)
            else:
                pass

        # creating dataframes for categorical and numerical dataframes
        colname = x.columns[cat]
        categoricalDataframe = x[colname].copy()
        colname1 = x.columns[num]
        numericalDataframe = x[colname1].copy()
        x_num = numericalDataframe.values  # array value of numericalDataframe
        x_cat = categoricalDataframe.values  # array value of categoricalDataframe

        if cat:
            # handling missing values in categoricalDataframe
            imputer1 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer1.fit(x_cat)
            x_cat = imputer1.transform(x_cat)

            # encoding x_cat
            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(), list(range(len(categoricalDataframe.columns))))],
                remainder='passthrough')
            x_cat = np.array(ct.fit_transform(x_cat))
            with open('kmeansencoder.pickle', 'wb') as f:
                pickle.dump(ct, f)

        if num:
            # handling missing values in numericalDataframe
            imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer1.fit(x_num)
            x_num = imputer1.transform(x_num)

        if cat != [] and num != []:
            t = pd.concat([pd.DataFrame(x_cat), pd.DataFrame(x_num)], axis=1, join='inner')
        elif cat != [] and num == []:
            t = pd.DataFrame(x_cat)
        elif cat == [] and num != []:
            t = pd.DataFrame(x_num)
        # t = pd.concat([pd.DataFrame(x_cat), pd.DataFrame(x_num)], axis=1, join='inner')

        KMeansResult = kMeansUnsupervised(t, 0, '', [])
        return [[0, KMeansResult, dtypeArray], allColumnsNamesInitially,
                countOfDatapoints, dict_from_list]  # [0,[x,x.shape],dtypeArray]

    # supervised learning
    else:
        # positioning the y column to the last of the dataframe

        allColumnsNamesInitially = dataset.columns.values.tolist()

        allIndependantColumnsNames = [allColumnsNamesInitially[yColumnNumber]]

        allColumnsNamesInitially1 = allColumnsNamesInitially
        allColumnsNamesInitially1.pop(yColumnNumber)
        allDependantColumnsNames = allColumnsNamesInitially1

        countOfDatapoints = len(dataset.index)

        cols = dataset.columns.tolist()
        cols = cols[:yColumnNumber] + cols[1 + yColumnNumber:] + cols[yColumnNumber:yColumnNumber + 1]
        dataset = dataset[cols]

        # getting x and y
        x = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1].values

        # getting array whose indexes contains values (
        #         # 1 ->  int64 categorical
        #         # 2 ->  int64 numerical
        #         # 3 ->  float64 (numerical)
        #         # 4 ->  string(object) with <2 gap (categorical)
        #         # 5 ->  string(object) with >=2gap (remove the column)
        #         )
        dtypeArray = finddtype(x)

        dictabc = {1: 'int64 dtype --> Numerical',
                   2: 'int64 dtype --> Categorical',
                   3: 'float64 dtype --> Numerical',
                   4: 'string type --> Categorical',
                   5: 'Long String type --> Removed'}
        newDtypeArray = []
        for i in dtypeArray:
            newDtypeArray.append(dictabc[i])
        dict_from_list = dict(zip(allDependantColumnsNames, newDtypeArray))


        # cat is array that contains column number of categorical columns in the dataset
        # num is array that contains column number of numerical columns in the dataset
        cat = []
        num = []
        for i in range(len(dtypeArray)):
            if dtypeArray[i] == 5:
                pass
            elif dtypeArray[i] in [1, 4]:
                cat.append(i)
            elif dtypeArray[i] in [2, 3]:
                num.append(i)
            else:
                pass

        # creating dataframes for categorical and numerical dataframes
        colname = x.columns[cat]
        categoricalDataframe = x[colname].copy()
        colname1 = x.columns[num]
        numericalDataframe = x[colname1].copy()
        x_num = numericalDataframe.values  # array value of numericalDataframe
        x_cat = categoricalDataframe.values  # array value of categoricalDataframe

        if cat:
            # handling missing values in categoricalDataframe
            imputer1 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputer1.fit(x_cat)
            x_cat = imputer1.transform(x_cat)

            # encoding x_cat
            ct = ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(), list(range(len(categoricalDataframe.columns))))],
                remainder='passthrough')
            x_cat = np.array(ct.fit_transform(x_cat))
            with open('supencoder.pickle', 'wb') as f:
                pickle.dump(ct, f)

        if num:
            # handling missing values in numericalDataframe
            imputer1 = SimpleImputer(missing_values=np.nan, strategy='median')
            imputer1.fit(x_num)
            x_num = imputer1.transform(x_num)

            # standardization of x_num
            sc = StandardScaler()
            x_num[:, :] = sc.fit_transform(x_num[:, :])
            with open('supstandardizer.pickle', 'wb') as f:
                pickle.dump(sc, f)

        t = pd.concat([pd.DataFrame(x_cat), pd.DataFrame(x_num)], axis=1, join='inner')

        ydf = pd.DataFrame(y)
        dtypeArrayofY = finddtype(ydf)  # array for finding if y is categorical(1,4) or numerical(2,3)
        dict_from_list1 = dict(zip(allIndependantColumnsNames, [dictabc[dtypeArrayofY[0]]]))

        if dtypeArrayofY[0] == 5:  # ERROR
            pass
            # return dtypeArrayofY
        elif dtypeArrayofY[0] in [1, 4]:  # apply classification   1
            le = LabelEncoder()
            ydf = le.fit_transform(ydf.values)
            svmResults = SVMclassification(t.values, ydf)  # svmResults==[Confusion_Matrix, Accuracy_Score]
            dtResults = decisionTreeClassification(t.values, ydf)  # dtResults==[Confusion_Matrix, Accuracy_Score]
            if svmResults[1] > dtResults[1]:
                dtypeArrayofY += svmResults
                dtypeArrayofY.append('Support Vector Machine Classification Model')
            else:
                dtypeArrayofY += dtResults
                dtypeArrayofY.append('Decision Tree Classification Model')

        elif dtypeArrayofY[0] in [2, 3]:  # apply regression  2
            lrRMSEaccuracy = linearRegression(t.values, ydf.values, [], dtypeArray)
            dtypeArrayofY += lrRMSEaccuracy
            dtypeArrayofY.append(dtypeArray)

        return [dtypeArrayofY, allColumnsNamesInitially, countOfDatapoints, dict_from_list, dict_from_list1]
