'''
Creator: Nirvan S P Theethira
Date: 04/24/2020
Purpose:  CSCI 5922 Spring Group Project: GeoGuessrLSTM
'''


from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
from math import sin, cos, sqrt, atan2, radians

# !pip install gmaps
import shapely
import pickle
import random
import numpy as np
import gmaps, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from ipywidgets.embed import embed_minimal_html
import webbrowser
import argparse

from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    '''
    Custom callback used to save model every few epochs.
    '''
    def __init__(self, gGuessr, saveFolder, modelNumber):
        super(LossAndErrorPrintingCallback, self).__init__()
        self.gGuessr = gGuessr
        self.modelNumber =  modelNumber
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        '''
        Save model every few epochs
        '''
        self.gGuessr.accuracy = round(float(logs['loss']),3)
        # if epoch%self.saveEpoch==0:
        #     self.gGuessr.save(self.saveFolder)

    def on_train_end(self, logs={}):
        '''
        Save model at the end of training
        '''
        print("Training sucessfull!!")
        self.gGuessr.save(self.saveFolder, self.modelNumber)

class Geoguessr:
    def __init__(self, model=None, accuracy=-1, useRestnet = True, 
                 inputShape=(3, 300, 600, 3), gridCount=243,
                 modelOptimizer=tf.keras.optimizers.Adam()):
        if model==None:
            convnet = tf.keras.Sequential()
            if useRestnet:
                restnet = ResNet50(include_top=False, weights='imagenet', input_shape=inputShape[1:])
                restnet.trainable = False
                convnet.add(restnet)
            else:
                convnet.add(Conv2D(64, (3,3), input_shape=inputShape[1:],
                    padding='same', activation='relu'))
                convnet.add(Conv2D(64, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.9))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(128, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(128, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.9))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(256, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(256, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.9))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(512, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(512, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.9))
            convnet.add(GlobalMaxPool2D())
            self.model = tf.keras.Sequential()
            self.model.add(TimeDistributed(convnet, input_shape=inputShape))
            self.model.add(LSTM(64))
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(gridCount, activation='softmax'))
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy, 
                      optimizer=modelOptimizer, metrics=['categorical_accuracy'])
        else:
            self.model = model
            
        self.model.summary()
        self.accuracy = accuracy
        
    def readData(self, fileNames, dataDir):
        numClasses = self.model.layers[-1].output_shape[-1]
        inputShape = self.model.layers[0].input_shape[2:4]
        X = np.array(list(map(lambda x: [np.array(load_img(dataDir+x+'/'+i, 
                                                           target_size=inputShape)) 
                                         for i in os.listdir(dataDir+x)], 
                              fileNames)))
        
        y = tf.keras.utils.to_categorical(list(map(lambda x:int(x.split('+')[0]), fileNames)), 
                                  num_classes=numClasses)
        return X,y
    
    def dataGen(self, fileNames, dataDir, batchSize=10, infinite=True):
        totalBatches = len(fileNames)/batchSize
        counter=0
        while(True):
            prev = batchSize*counter
            nxt = batchSize*(counter+1)
            counter+=1
            yield self.readData(fileNames[prev:nxt],dataDir)
            if counter>=totalBatches:
                if infinite:
                    counter=0
                else:
                    break
    
    def fit(self, trainFiles, dataDir, saveFolder, batchSize = 10, 
            epochs = 20, 
            plot=False):
        # list of image file names
        # eg: <gridNo>+<lat,long>+<imageNo_date>.jpg 
        # eg: 60+48.4271513,-110.5611851+0_2009-06.jpg
        print("Getting data from directory: {}".format(dataDir))
        accuracy = []
        loss = []
        cnt = 0
        for X,y in self.dataGen(trainFiles, dataDir, batchSize=batchSize, infinite=False):
            callBack = [LossAndErrorPrintingCallback(self, saveFolder, cnt)]
            print("Read {} points. Training now".format(len(X)))
            evalutaion = self.model.fit(X,y,
                                        epochs=epochs, steps_per_epoch = len(X),
                                        callbacks=callBack)
            accuracy += evalutaion.history['categorical_accuracy']
            loss += evalutaion.history['loss']
            cnt += 1
        if plot:
            plt.plot(accuracy)
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.show()

            plt.plot(loss)
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()
            
    def save(self, saveFolder, modelNumber=0):
        if self.accuracy==-1:
            print("Cannot save untrained model!!!")
        else:
            print("\nSaving model {} with loss {} at {}".format(modelNumber,
                                                                self.accuracy, 
                                                               saveFolder))
            self.model.save(saveFolder + '/model_{}_{}.h5'.format(self.accuracy,
                                                                  modelNumber))

    @classmethod
    def load(cls, loadFile):
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        accuracy = float(modelFile.split('_')[1])
        print("Loaded model loss {}".format(accuracy))
        return cls(model=model, accuracy=accuracy)
    
    def haversine(self, lati1, long1, lati2, long2):
        # approximate radius of earth in miles
        R = 3958.8

        lat1 = radians(lati1)
        lon1 = radians(long1)
        lat2 = radians(lati2)
        lon2 = radians(long2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c
    
    def gridDist(self, gridPoly1, gridPoly2):
        c1 = Polygon(np.flip(gridPoly1)).centroid
        lati1, long1 = c1.y, c1.x
        c2 = Polygon(np.flip(gridPoly2)).centroid
        lati2, long2 = c2.y, c2.x
        h = self.haversine(lati1, long1, lati2, long2)
        return h, [lati1, long1], [lati2, long2]
    
    def evaluate(self, imgFiles, dataDir, ployGrid, checkPoint=50):
        dists = []
        ln = len(imgFiles)
        for idx,(xx,yy) in enumerate(self.dataGen(imgFiles, dataDir, batchSize=1, infinite=False)):
            yp = self.model.predict(xx)[0]
            yn = list(map(lambda x:x/max(yp), yp))
            dist, _, _ = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
            dists.append(dist)
            if idx%checkPoint==0:
                print("Evaluated {} out of {} points".format(idx, ln))
        return np.average(dists)
            
    def predictSingle(self, imgFile, dataDir, ployGrid=None):
        xx,yy = self.readData([imgFile], dataDir)
        yp = self.model.predict(xx)[0]
        yn = list(map(lambda x:x/max(yp), yp))
        dist, start, end = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
        if ployGrid:
            mx = max(yn)
            mn = min(yn)
            plt.plot([start[1],end[1]], [start[0],end[0]], color='black', 
                     label="Distance: {} miles".format(round(dist,3)))
            for k,i in ployGrid.items():
                if k==np.argmax(yy[0]):
                    plt.plot(i[:,1],i[:,0],color='blue',label="Actual Grid", alpha=1)
                else:
                    plt.plot(i[:,1],i[:,0],color='black', alpha=0.7)
                plt.fill(i[:,1],i[:,0],color='red', alpha=yn[k])
            plt.legend(loc="lower left")
            plt.show()
            
            gPoly = []
            gLine = gmaps.Line(
                start=start,
                end=end,
                stroke_color = 'blue'
            )
            for grid, polygon in ployGrid.items():
                gPoly.append(gmaps.Polygon(
                                        list(polygon),
                                        stroke_color='black',
                                        fill_color='red',#rgb(mn,mx,float(yn[grid])),
                                        fill_opacity=float(yn[grid])
                                        ))
            fig = gmaps.figure(center=(39.50,-98.35), zoom_level=4)
            fig.add_layer(gmaps.drawing_layer(features=gPoly))
            fig.add_layer(gmaps.drawing_layer(features=[gLine]))
            fig.add_layer(gmaps.symbol_layer([start], scale=3, 
                                 fill_color='green',stroke_color='green', info_box_content='Expected'))
            fig.add_layer(gmaps.symbol_layer([end], scale=3, 
                                             fill_color='yellow', stroke_color='yellow', 
                                             info_box_content='Predicted: {}'.format(dist)))
            embed_minimal_html('gmap.html', views=fig)
            webbrowser.open('gmap.html',new=1)
        return dist