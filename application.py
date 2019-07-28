from flask import Flask, render_template, request
import os
from datetime import datetime, timedelta
import sqlite3 as sql
import random
import time
import sqlite3
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import numpy as np
from matplotlib import style
import pandas as pd
from sklearn import cluster
application = Flask(__name__)



@application.route('/')
def home():
   return render_template('home.html')

@application.route('/cluster1',methods = ['POST', 'GET'])
def cluster1():
    dir_name = "static/"
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".png"):
            os.remove(os.path.join(dir_name, item))
    pyplot.clf()
    num = int(request.form['num'])
    xaxis = str(request.form['x'])
    yaxis = str(request.form['y'])
    conn = sqlite3.connect('database.db')
    qry = "select " + xaxis + ", " + yaxis + " from titanic3 where " + xaxis + " is not '' and " + yaxis + " is not ''"
    df = pd.read_sql(qry, conn)
    X = df.values #returns a numpy array
    k = num
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    cl = []
    for i in range(k):
        ds = X[np.where(labels==i)]
        cl.append(ds)
        pyplot.plot(ds[:,0],ds[:,1],'o')
        lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
        pyplot.setp(lines,ms=15.0)
        pyplot.setp(lines,mew=2.0)
    # pyplot.show()
    fname = time.strftime("%Y%m%d-%H%M%S") + ".png"
    pic = "static/" + fname
    pyplot.suptitle(xaxis.capitalize() + " VS " + yaxis.capitalize(), fontsize=20)
    pyplot.xlabel(xaxis.capitalize(), fontsize=18)
    pyplot.ylabel(yaxis.capitalize(), fontsize=16)
    pyplot.savefig(pic)
    data = []
    centDist = []
    for i in range(len(cl)):
        temp = []
        print("Cluster # " + str(i))
        cl[i] = cl[i].tolist()
        cen = []
        for j in range(0,len(cl[i])):
            cen.append(np.linalg.norm(cl[i][j]-centroids[i]))
            temp2 = {}
            temp2['survive'] = cl[i][j][0]
            temp2['fare'] = cl[i][j][1]
            temp.append(temp2)
#            print(temp2)
        data.append(temp)
        centDist.append(cen)
    exp = []
    for i in range(len(centDist)):
        temp = {}
        temp['clusterNumber'] = i + 1
        temp['numElements'] = len(centDist[i])
        centDist[i] = max(centDist[i])
        temp['diameter'] = round(centDist[i] * 2, 2)
        temp['x'] = round(centroids[i][0], 2)
        temp['y'] = round(centroids[i][1], 2)
        print(temp)
        exp.append(temp)
    return render_template('cluster1.html', nm = num, data=data, pic = pic,  xaxis = xaxis.capitalize(), yaxis = yaxis.capitalize(), exp = exp)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
