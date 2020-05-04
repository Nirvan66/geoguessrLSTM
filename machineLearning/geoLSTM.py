'''
Creator: Nirvan S P Theethira
Date: 04/24/2020
Purpose:  CSCI 5922 Spring Group Project: GeoGuessrLSTM

NOTE: Make sure the commands are run in the `geoguessrLSTM/machineLearning/` directory

SAMPLE TRAIN RUN
python geoLSTM.py

SAMPLE TEST RUN
python geoLSTM.py --testModelName restnet_5.738_19.h5
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

# directory containing data, train test file split
DATADIR = "../infoExtraction/data"
# directory containing combined location data folders
DATACOMBINED = DATADIR + "/dataCombinedSamples/"
# directory containing polygon gird shapes
POLYDIR = "../infoExtraction" 
# directory to store/load model in/from
MODELDIR = "models"

# NOTE: The data found in these locations are sample. Look at the drive for all data

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    '''
    Custom callback used to save model at the end of a training batch.
    '''
    def __init__(self, gGuessr, saveFolder, modelNumber):
        '''
        gGuess: Instance of Geoguessr
        saveFolder: Name of floder to save to
        modelNumber: A number to add to saved model file name. 
        Used to find how far into training a model was saved
        '''
        super(LossAndErrorPrintingCallback, self).__init__()
        self.gGuessr = gGuessr
        self.modelNumber =  modelNumber
        self.saveFolder = saveFolder

    def on_epoch_end(self, epoch, logs=None):
        '''
        Update model loss every few epochs
        '''
        self.gGuessr.loss = round(float(logs['loss']),3)

    def on_train_end(self, logs={}):
        '''
        Save model at the end of training
        '''
        print("Training sucessfull!!")
        self.gGuessr.save(self.saveFolder, self.modelNumber)

class Geoguessr:
    '''
    The class has all the functions required to built, train and test the geoguessr LSTM model.
    '''
    def __init__(self, model=None, loss=-1, useRestnet = True, 
                 inputShape=(3, 300, 600, 3), gridCount=243,
                 modelOptimizer=tf.keras.optimizers.Adam()):
        '''
        The function is used to load or initialize a new model
        useRestnet : set to True to use pretrained frozen restnet model
                     set to False to use trainable CNN model
        inputShape: Shape of input image set 
                    (<numer-of-images>, <image-width>, <image-height>, <RGB-values>)
        gridCount: Number of ouput grids to predict on
        '''
        if model==None:
            convnet = tf.keras.Sequential()
            if useRestnet:
                # use restnet CNN
                restnet = ResNet50(include_top=False, weights='imagenet', input_shape=inputShape[1:])
                # Freeze model
                restnet.trainable = False
                convnet.add(restnet)
            else:
                # Use trainable CNN
                convnet.add(Conv2D(128, (3,3), input_shape=inputShape[1:],
                    padding='same', activation='relu'))
                convnet.add(Conv2D(128, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.6))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(64, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(64, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.6))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(64, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(64, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.6))
                convnet.add(MaxPool2D())
                convnet.add(Conv2D(512, (3,3), padding='same', activation='relu'))
                convnet.add(Conv2D(512, (3,3), padding='same', activation='relu'))
                convnet.add(BatchNormalization(momentum=.6))
            convnet.add(GlobalMaxPool2D())
            self.model = tf.keras.Sequential()
            # Connect the CNN to an LSTM
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
            # load pre trained model
            self.model = model
            
        self.model.summary()
        self.loss = loss
        
    def readData(self, fileNames, dataDir):
        '''
        Takes a list of location folder names and ouputs a list of input image vectors and ouput categorical grid vector pairs.
        fileNames should look like: 60+48.4271513,-110.5611851
        '''
        numClasses = self.model.layers[-1].output_shape[-1]
        inputShape = self.model.layers[0].input_shape[2:4]
        # load three images in a folder as arrays 
        X = np.array(list(map(lambda x: [np.array(load_img(dataDir+x+'/'+i, 
                                                           target_size=inputShape)) 
                                         for i in os.listdir(dataDir+x)], 
                              fileNames)))
        # load grid numbers from folder names and convert them to categorical ouput vectors
        y = tf.keras.utils.to_categorical(list(map(lambda x:int(x.split('+')[0]), fileNames)), 
                                  num_classes=numClasses)
        return X,y
    
    def dataGen(self, fileNames, dataDir, batchSize=10, infinite=True):
        '''
        Takes a list of location folder names and ouputs 
        a list of input image vectors and ouput categorical grid vector pairs in batches.
        The function is essentially used as a generator that calls readData in datches.
        Inifinit: Tells the function to stop or keep going once the list of file names has been iterated through
        fileNames should look like: 60+48.4271513,-110.5611851
        '''
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
        '''
        Used to train the model in datches with each batch going through a fixed number of epochs
        this is done to let the model have sufficient training time on each batch of data.

        trainFiles: list of image file names, eg: <gridNo>+<lat,long>, eg: 60+48.4271513,-110.5611851
        dataDir: Directory that stores combined image files eg: "/dataCombinedSamples/"
        saveFolder: Folder to save trained model to eg: "models"
        '''
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
        '''
        Saves model to specified folder with specified number along with loss
        saveFolder: Folder to save trained model to eg: "models"
        '''
        if self.loss==-1:
            print("Cannot save untrained model!!!")
        else:
            print("\nSaving model {} with loss {} at {}".format(modelNumber,
                                                                self.loss, 
                                                               saveFolder))
            self.model.save(saveFolder + '/model_{}_{}.h5'.format(self.loss,
                                                                  modelNumber))

    @classmethod
    def load(cls, loadFile):
        '''
        Loads model from specified folder with loss
        loadFile: file to load model from eg: "models/restnet_5.738_19.h5"
        '''
        print("Loading model from {}".format(loadFile))
        model = tf.keras.models.load_model(loadFile)
        modelFile = loadFile.split('/')[-1]
        loss = float(modelFile.split('_')[1])
        print("Loaded model loss {}".format(loss))
        return cls(model=model, loss=loss)
    
    def haversine(self, lati1, long1, lati2, long2):
        '''
        Gives distance in miles between two points on the planet specified by latitudes and longitudes
        lati1, long1: latitude and longitude of first location
        lati2, long2: latitude and longitude of second location
        https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        '''
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
        '''
        Gives distance in miles between the centers of two grids or polygons. 
        gridPoly1: first grid polygon. list of latitude and longitudes
        gridPoly2: second grid polygon. list of latitude and longitudes
        '''
        c1 = Polygon(np.flip(gridPoly1)).centroid
        lati1, long1 = c1.y, c1.x
        c2 = Polygon(np.flip(gridPoly2)).centroid
        lati2, long2 = c2.y, c2.x
        h = self.haversine(lati1, long1, lati2, long2)
        return h, [lati1, long1], [lati2, long2]
    
    def evaluate(self, imgFiles, dataDir, ployGrid, checkPoint=50):
        '''
        Calculates average of distances between target and predicted grids for a list of files
        imgFiles: List of test location image triplet folders. Each element of the list has to look like: 
        eg: <gridNo>+<lat,long>
        eg: 60+48.4271513,-110.5611851
        dataDir: Directory that stores combined image files eg: "/dataCombinedSamples/"
        polyGrid: List of polygons that contain make up the USA split into grids.
                  It can be loaded from eg: "infoExtraction/usaPolyGrid.pkl"
        checkPoint: report progress
        '''
        dists = []
        ln = len(imgFiles)
        for idx,(xx,yy) in enumerate(self.dataGen(imgFiles, dataDir, batchSize=1, infinite=False)):
            yp = self.model.predict(xx)[0]
            # evaluate distance for single point
            dist, _, _ = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
            dists.append(dist)
            if idx%checkPoint==0:
                print("Evaluated {} out of {} points".format(idx, ln))
        # take average of all distances
        return np.average(dists)
            
    def predictSingle(self, imgFile, dataDir, ployGrid):
        '''
        Predicts softmax ouput by trained model for single image and plots it 
        mgFiles: String that contains test location image triplet folder name. String has to look like:
        # <gridNo>+<lat,long>
        # 60+48.4271513,-110.5611851
        dataDir: Directory that stores combined image files eg: "/dataCombinedSamples/"
        polyGrid: List of polygons that contain make up the USA split into grids.
                  It can be loaded from eg: "infoExtraction/usaPolyGrid.pkl"
        '''
        # read image triplets from single file
        xx,yy = self.readData([imgFile], dataDir)
        # predict single image triplet
        yp = self.model.predict(xx)[0]
        # normalize prediction for better visualization
        yn = list(map(lambda x:x/max(yp), yp))
        # evaluate distance for single point
        dist, start, end = self.gridDist(ployGrid[np.argmax(yy[0])],ployGrid[np.argmax(yp)])
        mx = max(yn)
        mn = min(yn)
        # plot result using matplotlib
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
        
        # plot result using google maps API
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
                                    fill_color='red',
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
        # save and display html page containing google maps API
        embed_minimal_html('gmap.html', views=fig)
        webbrowser.open('gmap.html',new=1)
        return dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a geguessr model')
    parser.add_argument('--testModelName', type=str, help='Name of model to run test on. Required for test mode')

    args = parser.parse_args()
    # testing mode is pretrained model name is provided
    if args.testModelName!=None:
        print("In testing mode.")
        geoModel = Geoguessr.load(MODELDIR + '/' + args.testModelName)
        TESF = np.load(DATADIR + '/testFiles.npy')
        usaPolyGrid = pickle.load(open(POLYDIR + "/usaPolyGrid.pkl",'rb'))
        geoModel.predictSingle(TESF[0], DATACOMBINED , ployGrid=usaPolyGrid)

    # if pretrained model is not provide train a new one and save
    else:
        print("In training mode. Training new model")
        TF = np.load(DATADIR + '/trainFiles.npy')
        geoModel = Geoguessr(useRestnet = True)
        geoModel.fit(trainFiles = TF, 
                      dataDir = DATACOMBINED, 
                      saveFolder = MODELDIR,
                      epochs=2,
                      batchSize=4,
                      plot=True
                     )