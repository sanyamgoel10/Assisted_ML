import eda
from werkzeug.utils import secure_filename
from flask import *
import os

app = Flask(__name__)

global sanarr, sanarr1, sanarr2, sanarr3


@app.route("/", methods=['POST', 'GET'])
def index():
    directory = os.getcwd()
    test = os.listdir(directory)
    for item in test:
        if item.endswith(".pickle"):
            os.remove(os.path.join(directory, item))
        if item.endswith(".csv"):
            os.remove(os.path.join(directory, item))
    directory1 = os.getcwd()+"\\static\\images"
    test1 = os.listdir(directory1)
    for item in test1:
        if item.endswith(".png"):
            os.remove(os.path.join(directory1, item))
    return render_template('fileupload.html')


@app.route("/analysisOutput", methods=["POST", "GET"])
def index1():
    if request.method == 'POST':
        if request.form["yLinear"] == '':
            yLinear = -1
        else:
            yLinear = int(request.form["yLinear"])
        if request.form["indexColumn"] == '':
            indexColumn = -1
        else:
            indexColumn = int(request.form["indexColumn"])
        csvFile = request.files["csvFile"]
        csvFile.save(secure_filename(csvFile.filename))
        a = eda.func(yLinear, csvFile.filename, indexColumn)  # [0,[x,x.shape],dtypeArray]
        if a[0][0] == 0:
            global sanarr, sanarr1, sanarr2
            sanarr = a[0][1][0]  # x
            sanarr1 = a[0][1][1]  # x.shape
            sanarr2 = a[0][2]  # dtypeArray
            return render_template('unsup.html',cna=a[1],ndd=a[2],cnvcd=a[3])
        elif a[0][0] in [1,4]:
            return render_template('supClas.html', cmatrix=a[0][1], Accuracy=a[0][2], bestmodelname=a[0][3],cna=a[1],ndd=a[2],dvwtrd=a[3],ivwtrd=a[4])
        elif a[0][0] in [2,3]:
            global sanarr3
            sanarr3 = a[0][3]
            return render_template('supReg.html', RMSE=a[0][1], Accuracy=a[0][2],cna=a[1],ndd=a[2],dvwtrd=a[3],ivwtrd=a[4])
        elif a[0][0] == 5:
            return render_template('ErrorByUser.html',cna=a[1],ndd=a[2],dvwtrd=a[3])



@app.route("/KMeansOutput", methods=["POST", "GET"])
def kms():
    if request.method == 'POST':
        if request.form['noClusters'] == '' or request.form['noClusters'] == '0':
            return render_template('unsupKmeansErrorClusters.html')
        else:
            numberOfClusters = int(request.form['noClusters'])
            global sanarr, sanarr1, sanarr2
            pqr = [sanarr, sanarr1]  # [x, x.shape]
            pqr1 = sanarr2  # dtypeArray
            ab = eda.kMeansUnsupervised(pqr, numberOfClusters, '', pqr1)  #[str(y_kmeans), str(wcssError), n(number of clusters)]
            global sanarr3
            sanarr3 = ab[2]  # or numberOfClusters
            return render_template('unsupKmeans.html', nCl=numberOfClusters, predictions=ab[0], wcssError=ab[1])


@app.route("/KMeansOutputPredictions", methods=["POST", "GET"])
def kmsPred():
    if request.method == 'POST':
        if request.form['kmeansUserInput'] == '':
            return render_template('unsupKmeansErrorPredictions.html')
        else:
            kmeansUserInputforPredictions = str(request.form['kmeansUserInput'])
            global sanarr, sanarr1, sanarr2
            pqr = [sanarr, sanarr1]  # [x, x.shape]
            pqr1 = sanarr2  # dtypeArray
            global sanarr3
            pqr2 = sanarr3  # n -> number of clusters
            ab = eda.kMeansUnsupervised(pqr, pqr2, kmeansUserInputforPredictions, pqr1)
            return render_template('unsupervisedKmeansPredictions.html', kmUIfP=kmeansUserInputforPredictions, predictions1=ab[0][0])


@app.route("/LnRsOutput", methods=["POST", "GET"])
def lnrsPred():
    if request.method == 'POST':
        if request.form['lnrPred'] == '':
            return render_template('supRegError.html')
        else:
            global sanarr3
            dtparr = sanarr3
            y_pred = request.form['lnrPred']
            ab = eda.linearRegression([], [], y_pred, dtparr)
            return render_template('RegPredResult.html', nCl=ab[0], nCl1=ab[1])


if __name__ == '__main__':
    app.run(debug=True)