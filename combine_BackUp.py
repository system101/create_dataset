import os,glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import imutils
import itertools
from itertools import permutations
from itertools import combinations
from PIL import ImageDraw, ImageFont
from PIL import Image as I

def get_img_object(img_dir):
    img = cv2.imread(img_dir,cv2.IMREAD_UNCHANGED) 
    b,g,r,a = cv2.split(img)
    return a

def save_image(img_dir, des, op, img):
    rename_image = image_dir.split('/')
    folder = des + '/'
    cv2.imwrite(folder + rename_image[len(rename_image)-1].split('.')[0]+'_'+op+'.png', img)


# @@@@@@@@@@@@@@@@@@@@@@@@@ Group 1 operations @@@@@@@@@@@@@@@@@@@@@@@@@

def morphology(img, m, n, op):
    kernel = np.ones((m,n),np.uint8)
    if (op == 'erosion'): # 0 - 3 
        return cv2.erode(img,kernel,iterations = 1)
    elif (op == 'dilation'): # 0 - 3 
        return cv2.dilate(img,kernel,iterations = 1)
    elif (op == 'opening'): # 1 - 3
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif (op == 'closing'): # 1 - 3 
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif (op == 'gradient'): # 1 - 3 (NO 1,1)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def hit_and_miss(img, kernels, kernel_number): # 2 binary kernel (10,136)
    n= kernel_combination[kernel_number]
    t1,t2,t3 = int(n[0]),int(n[1]),int(n[2])
    t4,t5,t6 = int(n[3]),int(n[4]),int(n[5])
    t7,t8,t9 = int(n[6]),int(n[7]),int(n[8])
    styled_kernel = np.array([[t1,t2,t3],
                              [t4,t5,t6],
                              [t7,t8,t9]]) 
    return cv2.morphologyEx(img, cv2.MORPH_HITMISS, styled_kernel)
    

# @@@@@@@@@@@@@@@@@@@@@@@@@ Group 2 operations @@@@@@@@@@@@@@@@@@@@@@@@@

def shearing(img,isShearingLeft):
    '''
    shear left range [0.6,0.9]
    shear right range [0.1,0.4]
    '''
    y, x = img.shape[:2]
    distance_from_origin = 0.6
    
    s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner 
    s_b = [0,x-1]                    # source bottom-left corner

    if(isShearingLeft): 
        d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner                            
        d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
    else:
        d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner                            
        d_b = [0,y-1]                                               # destination bottom-left corner


    src_points = np.float32([s_l, s_r, s_b])
    dst_points = np.float32([d_l, d_r, d_b])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (y,x))
    return img_output

# Zoom in/out with shape (28x28)
def Zoom(img, level):
    zoom_img = cv2.resize(img,None,fx=level,fy=level)
    x,y = zoom_img.shape
    if (level > 1):  # 1.1 - 1.4
        x1 = int((x-28)*0.5)
        y1 = int((y-28)*0.5)
        x2 = 28+x1
        y2 = 28+y1
        return zoom_img[y1:y2, x1:x2]
    else: # 0.6 - 0.9
        temp = np.zeros((28,28),dtype='int')
        x1 = int((28-x)*0.5)
        y1 = int((28-y)*0.5)
        x2 = x+x1
        y2 = y1
        x3 = x1
        y3 = y+y1
        x4 = x2
        y4 = y3

        row = 0
        col = 0
        test = np.zeros((x,y),dtype='int')
        for i in range(len(temp)):
            if (y1 <= i <= y3):    
                for k in range(len(temp[i])):
                    if(x1 <= k < x2):
                        temp[i][k] = zoom_img[row%x][col%y]
                        col += 1
                row += 1
        return temp
    
def dictionary(command):
    dictionary = {
        1:'erosion',
        2:'dilation',
        3:'opening',
        4:'closing',
        5:'gradient',
        6:'hit and miss',
        7:'shaering',
        8:'zoom',
    }
    return dictionary[command]

def dictionary_operaion(command):
    dictionary = {
        1:'-E',
        2:'-D',
        3:'-S',
        4:'-Z',
        5:'-G'
    }
    return dictionary[command]



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ WORKING AREA $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class Image():
    def __init__(self, images):
        self.images = images
        self.operation = '_'

    def set_operation(self, operation):
        self.operation = operation

    def get_operation(self):
        return self.operation

    def get_image(self):
        return self.images
    
class Erosion(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because dilation has 4 posibilities
        self.operation = 'e'
        self.generate_Erosion()
        
    def generate_Erosion(self):
        combination = []
        count = 0
        isZero = False
        
        if (self.isOriginal):
            for i in range(3):
                for k in range(3):
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif(i != 0 and k != 0): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = self.images.get_operation() + self.operation
                        gen_img = Image(cv2.erode(self.images.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
        else:
            for i in range(3):
                for k in range(3):
                    for letter in self.images:
                        if (not isZero):
                            isZero = True
                            continue
                            isZero = True
                        elif(i != 0 and k != 0): # (0,0) and (3,2) not included
                            kernel = np.ones((i,k),np.uint8)
                            image_name = letter.get_operation() + self.operation
                            gen_img = Image(cv2.erode(letter.get_image(),kernel,iterations = 1))
                            gen_img.set_operation(image_name+str(i)+str(k))
                            combination.append(gen_img)
                
        self.images = combination

class Dilation(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because dilation has 4 posibilities
        self.operation = 'd'
        self.generate_Dilation()
        
    def generate_Dilation(self):
        combination = []
        isZero = False
        if (self.isOriginal):
            for i in range(4):
                for k in range(3):
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif((i != 0 and k != 0) and not(i == 3 and k ==2)): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = self.images.get_operation() + self.operation
                        gen_img = Image(cv2.dilate(self.images.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
        else:
            for i in range(4):
                for k in range(3):
                    for letter in self.images:
                        if (not isZero):
                            isZero = True
                            continue
                            isZero = True
                        elif((i != 0 and k != 0) and not(i == 3 and k ==2)): # (0,0) and (3,2) not included
                            kernel = np.ones((i,k),np.uint8)
                            image_name = letter.get_operation() + self.operation
                            gen_img = Image(cv2.dilate(letter.get_image(),kernel,iterations = 1))
                            gen_img.set_operation(image_name+str(i)+str(k))
                            combination.append(gen_img)
                
        self.images = combination

class Zoom(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because zoom has 4 posibilities (2 IN and 2 OUT)
        self.operation = 'z'
        self.generate_Zoom()

    def generate_Zoom(self):
        combination = []
        zoom_in_out = [[1.1,1.2,1.3],[0.8,0.9]]

        if(self.isOriginal):
                # Run every zoom level and apply to all of the input images
            for in_out in range(2): 
                for zoom_level in zoom_in_out[in_out]:
                    level = zoom_level
                    zoom_img = cv2.resize(self.images.get_image(),None,fx=level,fy=level)
                    x,y = zoom_img.shape
                    image_name = self.images.get_operation() + self.operation
                    if (level > 1):  # 1.1 - 1.4
                        x1 = int((x-64)*0.5)
                        y1 = int((y-64)*0.5)
                        x2 = 64+x1
                        y2 = 64+y1
                        gen_img = Image(zoom_img[y1:y2, x1:x2])
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)

                    else: # 0.6 - 0.9
                        temp = np.zeros((64,64),dtype='int')
                        x1 = int((64-x)*0.5)
                        y1 = int((64-y)*0.5)
                        x2 = x+x1
                        y2 = y1
                        x3 = x1
                        y3 = y+y1
                        x4 = x2
                        y4 = y3

                        row = 0
                        col = 0
                        test = np.zeros((x,y),dtype='int')
                        for i in range(len(temp)):
                            if (y1 <= i <= y3):    
                                for k in range(len(temp[i])):
                                    if(x1 <= k < x2):
                                        temp[i][k] = zoom_img[row%x][col%y]
                                        col += 1
                                row += 1
                        gen_img = Image(temp)
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)
        else:
            for in_out in range(2): 
                for zoom_level in zoom_in_out[in_out]:
                    for letter in self.images:
                        level = zoom_level
                        zoom_img = cv2.resize(letter.get_image(),None,fx=level,fy=level)
                        x,y = zoom_img.shape
                        image_name = letter.get_operation() + self.operation
                        if (level > 1):  # 1.1 - 1.4
                            x1 = int((x-64)*0.5)
                            y1 = int((y-64)*0.5)
                            x2 = 64+x1
                            y2 = 64+y1
                            gen_img = Image(zoom_img[y1:y2, x1:x2])
                            gen_img.set_operation(image_name+str(int(zoom_level*10)))
                            combination.append(gen_img)

                        else: # 0.6 - 0.9
                            temp = np.zeros((64,64),dtype='int')
                            x1 = int((64-x)*0.5)
                            y1 = int((64-y)*0.5)
                            x2 = x+x1
                            y2 = y1
                            x3 = x1
                            y3 = y+y1
                            x4 = x2
                            y4 = y3

                            row = 0
                            col = 0
                            test = np.zeros((x,y),dtype='int')
                            for i in range(len(temp)):
                                if (y1 <= i <= y3):    
                                    for k in range(len(temp[i])):
                                        if(x1 <= k < x2):
                                            temp[i][k] = zoom_img[row%x][col%y]
                                            col += 1
                                    row += 1
                            gen_img = Image(temp)
                            gen_img.set_operation(image_name+str(int(zoom_level*10)))
                            combination.append(gen_img)
        
        self.images = combination # add all generated images to the newly created Image object



class Shearing(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because shearing has 8 posibilities (4 LEFT and 4 RIGHT)
        self.operation = 's'
        self.generate_Shearing()

    def generate_Shearing(self):
        '''
        shear left range [0.6,0.9]
        shear right range [0.1,0.4]
        '''
        combination = []
        leftRight = [0,1]
        left_and_right_shear = [[0.1,0.2,0.3,0.4],[0.6,0.7,0.8,0.9]] # right and then left
        if (self.isOriginal):
            for isShearingLeft in leftRight: #[0,1]
                for angle in left_and_right_shear[isShearingLeft]:
                    img = self.images.get_image()
                    y, x = img.shape[:2]
                    distance_from_origin = angle
                    s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner 
                    s_b = [0,x-1]                    # source bottom-left corner
                    image_name = self.images.get_operation() + self.operation
                    
                    if(isShearingLeft):
                        d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner                            
                        d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
                    else:
                        d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner                            
                        d_b = [0,y-1]                                               # destination bottom-left corner

                    src_points = np.float32([s_l, s_r, s_b])
                    dst_points = np.float32([d_l, d_r, d_b])
                    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                    img_output = cv2.warpAffine(img, affine_matrix, (y,x))
                    gen_img = Image(img_output)
                    gen_img.set_operation(image_name+str(int(angle*10)))
                    combination.append(gen_img)
        else:
            for image in self.images: # loop to all the input images
                for isShearingLeft in leftRight: #[0,1]
                    for angle in left_and_right_shear[isShearingLeft]:
                        img = image.get_image()
                        y, x = img.shape[:2]
                        distance_from_origin = angle
                        s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner 
                        s_b = [0,x-1]                    # source bottom-left corner
                        image_name = image.get_operation() + self.operation
                    
                        if(isShearingLeft):
                            d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner                            
                            d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
                        else:
                            d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner                            
                            d_b = [0,y-1]                                               # destination bottom-left corner

                        src_points = np.float32([s_l, s_r, s_b])
                        dst_points = np.float32([d_l, d_r, d_b])
                        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                        img_output = cv2.warpAffine(img, affine_matrix, (y,x))
                        gen_img = Image(img_output)
                        gen_img.set_operation(image_name+str(int(angle*10)))
                        combination.append(gen_img)

        print('combination: ',len(combination))
        self.images = combination # add all generated images to the newly created Image object

class GaussianBlur(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because Gaussian blur has 2 posibilities (3x3 amd 1x3 kernel)

        #print("Gauss[IN]: ", self.images[0].get_image().dtype)
        self.operation = 'g'
        self.generate_GaussianBlur()

    def generate_GaussianBlur(self):
        combination = []
        if(self.isOriginal):
            img = self.images.get_image().astype(np.uint8)
            for k in range(2):
                image_name = self.images.get_operation() + self.operation
                if (k == 0):
                    gen_img = Image(cv2.GaussianBlur(img,(3,3),0))
                    gen_img.set_operation(image_name+'g')
                else:
                    gen_img = Image(cv2.blur(img,(1,3)))
                    gen_img.set_operation(image_name+'b')
                combination.append(gen_img)
        else:
            for letter in self.images: # apply hit and miss to all the input images
                img = letter.get_image().astype(np.uint8)
                for k in range(2):
                    image_name = letter.get_operation() + self.operation
                    if (k == 0):
                        gen_img = Image(cv2.GaussianBlur(img,(3,3),0))
                        gen_img.set_operation(image_name+'g')
                    else:
                        gen_img = Image(cv2.blur(img,(1,3)))
                        gen_img.set_operation(image_name+'b')
                    combination.append(gen_img)
        self.images = combination # add all generated images to the newly created Image object

def save_images(letter, des, img):
    folder = des + '/'
    for element in img: # save all the images
        cv2.imwrite(folder + letter + element.get_operation() +'.png', element.get_image())

def generate_permutaion_operation():
    dic = {
    '1':4,
    '2':5,
    '3':8,
    '4':5,
    '5':2,
    }

    s = 0
    
    summ = 0
    stuff = list(range(1,6))
    permutation_operations = []
##    print('stuff: ',stuff)
    for i in range(1,6):
        perm = combinations(stuff,i)#permutations
        c = 0
        s2 = 0
        for i in list(perm):
            c += 1
            permutation_operations.append(i)
            s1 = 1
            for p in i:
                s1 *= dic[str(p)]
            s2 += s1
        summ += s2
        s += c

        

    '''

    OPERATION (unit in generated image)
    (1)Erosion = 5
    (2)Dilation = 5
    (3)Shearing(left) = 3
       Shearing(right) = 3
    (4)Zoom(in) = 2
       Zoom(out) = 2
    (5)Gaussian Blur = 2
    =>Total operation: 5291

    Total training dataset: 5291 * 26 = 137566 

    '''
##    print(s)
    print('summ: ',summ)
    return permutation_operations


def recursive_operation(op, pointer, imgs):
    if(pointer == len(op)):
        return imgs
    elif(op[pointer] == 1):
        return recursive_operation(op, pointer+1, Erosion(imgs).get_image())
    elif(op[pointer] == 2):
        return recursive_operation(op, pointer+1, Dilation(imgs).get_image())
    elif(op[pointer] == 3):
        return recursive_operation(op, pointer+1, Shearing(imgs).get_image())
    elif(op[pointer] == 4):
        return recursive_operation(op, pointer+1, Zoom(imgs).get_image())
    elif(op[pointer] == 5):
        return recursive_operation(op, pointer+1, GaussianBlur(imgs).get_image())

def text_phantom(font_dir, text, size):
    # Availability is platform dependent
    font = font_dir

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = I.new('L', [size, size], (255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 255] black background
    return 255-np.asarray(canvas)

def get_alphabets(file_dir):
  alphabets = []
  for i in range(97,123): # 'a' to 'z'
    alphabets.append(text_phantom(file_dir,chr(i), 64))
  return np.asarray(alphabets)

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

def run():
    import time
    operations = generate_permutaion_operation()
    font = 'black-and-bitter-coffee/Black-and-Bitter-Coffee-TTF'
    alphabets = get_alphabets(font)
    train = [] # for images
    train_labels = [] # for labels
    start_time = time.time()
    for permute in operations: # each permutation
        print(permute)
        for character in range(97,123): # 'a' to 'z'
            img = alphabets[character-97]
            image_object = Image(img)

            processed_image = recursive_operation(permute, 0, image_object)
            print(len(processed_image))
            for im in processed_image:
                train.append (im.get_image())
                train_labels.append((character-97)+1) # label 'a' = 1, 'z' = 26 
            #save_images(chr(character), destination_folder+'/'+chr(character),processed_image)
            
    # save numpy array as .npz formats
    print(train[90990])
    print(train[90990][0])
    print(len(train))
    a = plt.imshow(train[90990],cmap='gray')
    plt.show()
    train = np.array(train,dtype=np.uint8) #as emnist
    train_labels = np.array(train_labels,dtype=np.uint8) #as emnist
    np.savez('my_EMNIST',train,train_labels)
    print('FILES SAVED!!')
    end_time = time.time()
    print('Time used(minutes): ', (end_time-start_time)/60)

def test():
    x,y = load_data('my_EMNIST.npz')
    num = 113234
    print(x.shape)
    print(y.shape)
    print(y[num])
    plt.imshow(x[num])
    plt.show()

def test_each():
    font = 'black-and-bitter-coffee/Black-and-Bitter-Coffee-TTF'
    alphabets = get_alphabets(font)
    img = alphabets[3]
    image_object = Image(img)
    processed_image = recursive_operation([1,2], 0, image_object)
    print(len(processed_image))
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ WORKING AREA $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if __name__ == '__main__':
    #run()
    test()
    #operations = generate_permutaion_operation()
    #test_each()

''' THE WORKING CODE (END) '''

    

            
            
    
