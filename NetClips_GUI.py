from tkinter import *
from tkinter import filedialog, messagebox
import cv2
import os
import re
import time
from math import floor
import random
from imutils import paths

import numpy as np
from PIL import Image, ImageTk
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Reshape, TimeDistributed, LSTM, Dense, Input, Lambda, Flatten, Dropout
#from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing import image
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
#from keras.callbacks import EarlyStopping
import keras

### Main:
window = Tk()
window.title("TVCONal NetClips")
window.configure(background ="white")
window.iconbitmap('TVCONal_logo.ico')

wd = './NetClips'
model_dir = './Models/'
model_name = 'sn24_resnet.h5'
model_cnn_lstm_name = 'lrcn_sn3_resnet.h5'
lstm_weights_name = 'model.h5'
model = load_model(model_dir + model_name)
model_cnn_lstm = load_model(model_dir + model_cnn_lstm_name)

def lrcn(Tx, num_activations, NUM_CLASSES, num_dims):
    X = Input(shape = (Tx, num_dims))
    a0 = Input(shape = (num_activations,))
    c0 = Input(shape = (num_activations,))

    a = a0
    c = c0

    outputs = []

    reshape_layer = Reshape((1,num_dims))
    lstm_layer = LSTM(num_activations, return_state = True)
    dense_layer = Dense(NUM_CLASSES, activation = 'softmax')

    for t in range(Tx):
        x = Lambda(lambda x: X[:,t,:])(X) # output shape will be (1, 1, num_dims)
        x = reshape_layer(x) # reshapes to (1, num_dims)
        a, _, c = lstm_layer(x, initial_state = [a, c])
        out = dense_layer(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs = outputs)
    opt = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_frames = 50    # per clip
num_activations = 64
NUM_CLASSES_lstm   = 4   # boundary 4, boundary 6, wicket, others
num_dims   = 2048  # per frame

lstm_model = lrcn(num_frames, num_activations, NUM_CLASSES_lstm, num_dims)
lstm_model.load_weights(model_dir + lstm_weights_name)

# Change path to your relevant directory

IMAGE_SIZE = (224, 224)
NUM_CLASSES  = 5
# BATCH_SIZE    = 20  # try reducing batch size or freeze more layers if your GPU runs out of memory
# FREEZE_LAYERS = 150  # freeze the first this many layers for training
# NUM_EPOCHS    = 20

temp_dir = './Temp/'
output_fileformat = '.avi'
tempframe_format = '.jpg'

def close_window():
    window.destroy()
    exit()

def browse():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global filename
    filename = filedialog.askopenfilename()
    vidPath_label.config(text=filename)
    print(filename)
    # e.g. filename = C:/Users/super/Desktop/Test.txt

def process():
    if not 'filename' in globals():
        messagebox.showerror("Processing Error", "No file detected: Please select valid file under Insert video clip to label.")
    else:
        labelsDisplay.delete(0.0, END)
        filedir = str(re.search('(.+/).+\.\w+',str(filename)).group(1))
        # get frames
        capture = cv2.VideoCapture(filename)
        count = 1
        class_count = [0,0,0,0]
        class_count_lstm = [0,0,0]
        # model = load_model(model_dir + model_name)
        startTime = time.time()
        while (capture.isOpened()):
            #starttime = time.time()
            ret, frame = capture.read()
            if ret:
                # Store frame on Temp directory and feed it as img_path
                img_path = temp_dir + 'tempframe-%d' % (count) + tempframe_format
                cv2.imwrite(img_path,frame)
                vidfile_label['text'] = 'Frame(s) processed: ' + str(count)
                count += 1
                #vidfile_label.config(text='Frame(s) processed: ' + str(count))
            else:
                capture.release()

        # Apply jittered sampling here
        allFrames = [frame for frame in os.listdir('./Temp') if tempframe_format in frame]

        imgidx = []
        for idx in range(len(allFrames)):
            imgidx.append(idx)
        totalFramesNum = len(imgidx)
        desiredNum = 50
        if desiredNum > totalFramesNum:
            print('Error Message: Clip requires a minimum of 50 frames to be processed.')

        else:
            jitSample = jitSampling(desiredNum,totalFramesNum) # e.g. [2,5,7,8,10,12.....]
        for num in imgidx:
            if num not in jitSample:
                os.remove('./Temp/' + allFrames[num])

        # Using model to predict each of the frames after jittered sampling
        allFrames = [file for file in os.listdir('./Temp') if tempframe_format in file]
        count = 1
        for frame in allFrames:
            img = image.load_img('./Temp/' + frame, target_size = IMAGE_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            result = model.predict(x, batch_size = 1, verbose = 1)

            result_cnn_lstm = model_cnn_lstm.predict(x, batch_size = 1, verbose = 1) # [1,2048] for each frmae


            # [[ChrisGayle GlennMaxwell Malinga Others ShaneWatson]]
            # max_idx = 0
            # max_value = result[0,0]
            # for idx in range(1,5):
            #     if result[0,idx] > max_value:
            #         max_value = result[0,idx]
            #         max_idx = idx
            # if max_idx == 0:
            #     print("ChrisGayle")
            # elif max_idx == 1:
            #     print("GlennMaxwell")
            # elif max_idx == 2:
            #     print("Malinga")
            # elif max_idx == 3:
            #     print("Others")
            # elif max_idx == 4:
            #     print("ShaneWatson")
            count += 1

            # Initializing resultArray
            if count == 2:
                resultArray = result
                resultArray_cnn_lstm = result_cnn_lstm
            elif count > 2:
                resultArray = np.append(resultArray,result, axis = 0)
                resultArray_cnn_lstm = np.append(resultArray_cnn_lstm,result_cnn_lstm, axis = 0)

        lstm_input = resultArray_cnn_lstm.reshape(1, 50, num_dims)
        a0 = np.zeros((1, num_activations))
        c0 = np.zeros((1, num_activations))
        result_lstm = lstm_model.predict([lstm_input ,a0, c0], batch_size = 50, verbose = 1)

        # Vector Aggregator for each video clip
        baseline_array = np.zeros(resultArray.shape)
        baseline_array.fill(1/NUM_CLASSES)

        diff_matrix = (resultArray - baseline_array).clip(min=0)
        epsilon = 1e-10 # a small epsilon is added to ensure no division by zero
        inverse_column_sum = np.reciprocal(diff_matrix.sum(axis = 0) + epsilon)
        weighted_matrix = np.multiply(inverse_column_sum, diff_matrix)
        score_matrix = np.multiply(weighted_matrix, resultArray)
        final_vector = score_matrix.sum(axis = 0)

        # Display final weights vector
        vidData_label.config(text='Final output: ' + str(final_vector))
        print(final_vector)
        MAX_IDX = 0
        MAX_VALUE = final_vector[0]
        for idx in range(1,5):
            if final_vector[idx] > MAX_VALUE:
                MAX_VALUE = final_vector[idx]
                MAX_IDX = idx
        if MAX_IDX == 0:
            print("ChrisGayle in Clip")
            if class_count[0] == 0:
                labelsDisplay.insert(END, 'ChrisGayle_')
                class_count[0] = 1
        elif MAX_IDX == 1:
            print("GlennMaxwell in Clip")
            if class_count[1] == 0:
                labelsDisplay.insert(END, 'GlennMaxwell_')
                class_count[1] = 1
        elif MAX_IDX == 2:
            print("Malinga in Clip")
            if class_count[2] == 0:
                labelsDisplay.insert(END, 'LasithMalinga_')
                class_count[2] = 1
        elif MAX_IDX == 3:
            print("Others")
        elif MAX_IDX == 4:
            print("ShaneWatson in Clip")
            if class_count[3] == 0:
                labelsDisplay.insert(END, 'ShaneWatson_')
                class_count[3] = 1

        resultArray_lstm = result_lstm[0]

        for array_index in range(1,len(result_lstm)):
            resultArray_lstm = np.append(resultArray_lstm, result_lstm[array_index])

        resultArray_lstm = resultArray_lstm.reshape(NUM_CLASSES_lstm,num_frames)

        # Vector Aggregator for each video clip
        baseline_array_lstm = np.zeros(resultArray_lstm.shape)
        baseline_array_lstm.fill(1/NUM_CLASSES)

        diff_matrix_lstm = (resultArray_lstm - baseline_array_lstm).clip(min=0)
        inverse_column_sum_lstm = np.reciprocal(diff_matrix_lstm.sum(axis = 0) + epsilon)
        weighted_matrix_lstm = np.multiply(inverse_column_sum_lstm, diff_matrix_lstm)
        score_matrix_lstm = np.multiply(weighted_matrix_lstm, resultArray_lstm)
        final_vector_lstm = score_matrix_lstm.sum(axis = 0)

        # Record time
        endTime = time.time()
        time_taken = endTime - startTime
        vidtime_label.config(text = 'Time taken: ' + str(round(time_taken,3)) + ' seconds')

        # Tagging
        MAX_IDX_lstm = 0
        MAX_VALUE_lstm = final_vector_lstm[0]
        for idx in range(1,4):
            if final_vector_lstm[idx] > MAX_VALUE_lstm:
                MAX_VALUE_lstm = final_vector_lstm[idx]
                MAX_IDX_lstm = idx
        if MAX_IDX_lstm == 0:
            print("ChrisGayle in Clip")
            if class_count_lstm[0] == 0:
                labelsDisplay.insert(END, 'Boundary_4')
                class_count_lstm[0] = 1
        elif MAX_IDX_lstm == 1:
            print("GlennMaxwell in Clip")
            if class_count_lstm[1] == 0:
                labelsDisplay.insert(END, 'Boundary_6')
                class_count_lstm[1] = 1
        elif MAX_IDX_lstm == 3:
            print("Malinga in Clip")
            if class_count_lstm[2] == 0:
                labelsDisplay.insert(END, 'Wicket')
                class_count_lstm[2] = 1
        elif MAX_IDX_lstm == 2:
            print("Others")

        for frames in allFrames:
            os.remove('./Temp/' + frames)

def tagVid():
    if len(str(labelsDisplay.get("1.0",'end-1c'))) == 0:
        messagebox.showerror("Naming Error", "No labels were identified: Please process video clip to generate valid labels.")
    else:
        # Need to get file directory
        filedir = str(re.search('(.+/).+\.\w+',str(filename)).group(1))
        prevName = str(re.search('.+/(.+\.\w+)',str(filename)).group(1))
        newName = filedir + str(labelsDisplay.get("1.0",'end-1c')) + output_fileformat
        os.rename(prevName,newName)

# Returns list of indexes of the selected files via jittered sampling
def jitSampling(desiredNum,totalFrames):
    selectionidx = []
    for iterRange in range(1,desiredNum+1):
        tempList = []
        if totalFrames % desiredNum == 0:
            for iterNum in range(floor(totalFrames/desiredNum)*(iterRange-1),floor(totalFrames/desiredNum)*(iterRange)):
                tempList.append(iterNum)
        else:
            for iterNum in range(floor(totalFrames/desiredNum)*(iterRange-1),floor(totalFrames/desiredNum)*(iterRange)):
                tempList.append(iterNum)
                if iterRange == desiredNum:
                    for extraNum in range(floor(totalFrames/desiredNum)*(iterNum),totalFrames+1):
                        tempList.append(extraNum)
        choice = random.choice(tempList)
        selectionidx.append(choice)
    return selectionidx

# Insert Company Logo
load_jpg = Image.open("TVCONal_logo.jpg")
load_jpg.resize((100,22))
logo_jpg = ImageTk.PhotoImage(load_jpg)
#logo = PhotoImage(file = 'TVCONal_logo.gif')
Label(window, image = logo_jpg, bg = 'white').grid(row = 0, column = 0, sticky = W)

### Create Label
insertVid_label = Label(window, text = "Insert video clip to label", bg = "white", fg = "black", font = "none 12 bold")
insertVid_label.grid(row = 2, column = 0, sticky = W)

### Add a browse button to search for file
Button(window, text = "Browse", width = 10, command = browse).grid(row = 3, column = 1, sticky = E)

### Create label box for file directory
vidPath_label = Label(window, bg = "white", fg = "black", font = "none 10",)
vidPath_label.grid(row = 3, column = 0, sticky = W)

### Add a process clip button to run on saved model
Button(window, text = "Process Clip", width = 12, command = process).grid(row = 4, column = 0, sticky = W)

### Label for vidData
insertVid_label = Label(window, text = "Processed Data:", bg = "white", fg = "black", font = "none 12 bold")
insertVid_label.grid(row = 6, column = 0, sticky = W)

### Add Label box for processed frame count
vidfile_label = Label(window, bg = "white", fg = "black", font = "none 10", text='Frame(s) processed: ')
vidfile_label.grid(row = 7, column = 0, sticky = W)
#vidfile_label.pack()

### Add Label box for processed frame count
vidtime_label = Label(window, bg = "white", fg = "black", font = "none 10", text = 'Time taken: ')
vidtime_label.grid(row = 8, column = 0, sticky = W)

### Add Label box for input file Final vector
vidData_label = Label(window, bg = "white", fg = "black", font = "none 10", text='Final output: ')
vidData_label.grid(row = 9, column = 0, sticky = W)

### Create Label
labelsIdentified_label = Label(window, text = "Label(s) Identified: ", bg = "white", fg = "black", font = "none 12 bold")
labelsIdentified_label.grid(row = 11, column = 0, sticky = W)

### Create Label displaying model tags model identified
labelsDisplay = Text(window, width = 60, height = 6, wrap = WORD, background = "white")
labelsDisplay.grid(row = 12, column = 0, columnspan = 2, sticky = W)

### Add identified tags to specified video clip
Button(window, text = "Add Tag(s)", width = 12, command = tagVid).grid(row = 13, column = 0, sticky = W)

### Exit Label
Label(window, text = "Click to exit:", bg = "white", fg = "black", font = "none 12 bold").grid(row =14, column = 0, sticky = W)

### Add exit Button
Button(window, text = "Exit", width = 14, command = close_window).grid(row = 15, column = 0, sticky = W)

### Run main loop
window.mainloop()
