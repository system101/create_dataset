import os
import cv2
import glob
import numpy as np





###Train data
##train = []
##train_labels = []
##files = glob.glob ("a/*.png") # your image path
##for myFile in files:
##    image = cv2.imread (myFile,cv2.IMREAD_UNCHANGED)
##    train.append (image)
##    train_labels.append(1)
##
##train = np.array(train,dtype=np.uint8) #as mnist
##train_labels = np.array(train_labels,dtype=np.uint8) #as mnist
##
### convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
### for example (120 * 40 * 40 * 3)-> (120 * 4800)
####train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]])
##
### save numpy array as .npz formats
##np.savez('my_EMNIST',train,train_labels)


##load_data('my_EMNIST.npz')



def load_data(file_dir):
    # returns: train_images , train_labels 
    data = np.load('my_EMNIST.npz')
    return data['arr_0'], data['arr_1'] 
    
def create_dataset(image_file_dir):
    file_dir = image_file_dir
    count_label = 0
    train = [] # for images
    train_labels = [] # for labels
    for dirpath, dirnames, filenames in os.walk(file_dir):
        if (count_label == 0):
            count_label += 1
            continue
        for image_name in filenames: # read each alphabet folder
            image_dir = dirpath+'/'+image_name
            image_dir = image_dir.replace('\\','/')
            #Train data
            files = glob.glob (image_dir) # image path
            image = cv2.imread (image_dir,cv2.IMREAD_UNCHANGED)
            train.append (image)
            train_labels.append(count_label)

        count_label += 1

    # save numpy array as .npz formats
    train = np.array(train,dtype=np.uint8) #as emnist
    train_labels = np.array(train_labels,dtype=np.uint8) #as emnist
    np.savez('my_EMNIST',train,train_labels)
    print('FILES SAVED!!')

    
##create_dataset('letters')  
x,y = load_data('my_EMNIST')
