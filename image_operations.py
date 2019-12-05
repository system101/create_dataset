import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import imutils
import itertools
from itertools import permutations
from itertools import combinations 

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

def combination(command):
    dictionary = {
        1:'erosion',
        2:'dilation',
        3:'shaering',
        4:'zoom'
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
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.images = [images] * 6 # because dilation has 6 posibilities
        self.operation = 'e'
        self.generate_Erosion()
        
    def generate_Erosion(self):
        combination = []
        count = 0
        isZero = False
        
        for i in range(4):
            for k in range(3):
                for letter in self.images:
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif((i != 0 and k != 0) and not(i == 3 and k == 2)): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = letter.get_operation() + self.operation
                        gen_img = Image(cv2.erode(letter.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
                
        self.images = combination

class Dilation(Image):
    def __init__(self, images):
        super().__init__(images)
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.images = [images] * 6 # because dilation has 6 posibilities
        self.operation = 'd'
        self.generate_Erosion()
        
    def generate_Erosion(self):
        combination = []
        isZero = False

        for i in range(4):
            for k in range(3):
                for letter in self.images:
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif((i != 0 and k != 0) and not(i == 3 and k == 2)): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = letter.get_operation() + self.operation
                        gen_img = Image(cv2.dilate(letter.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
                
        self.images = combination

class Zoom(Image):
    def __init__(self, images):
        super().__init__(images)
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.images = [images] * 6 # because zoom has 6 posibilities (3 IN and 3 OUT)
        self.operation = 'z'
        self.generate_Zoom()

    def generate_Zoom(self):
        combination = []
        zoom_in_out = [[1.1,1.2,1.3],[0.7,0.8,0.9]]

        # Run every zoom level and apply to all of the input images
        for in_out in range(2): 
            for zoom_level in zoom_in_out[in_out]:
                for letter in self.images:
                    level = zoom_level
                    zoom_img = cv2.resize(letter.get_image(),None,fx=level,fy=level)
                    x,y = zoom_img.shape
                    image_name = letter.get_operation() + self.operation
                    if (level > 1):  # 1.1 - 1.4
                        x1 = int((x-28)*0.5)
                        y1 = int((y-28)*0.5)
                        x2 = 28+x1
                        y2 = 28+y1
                        gen_img = Image(zoom_img[y1:y2, x1:x2])
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)

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
                        gen_img = Image(temp)
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)
        self.images = combination # add all generated images to the newly created Image object



class Shearing(Image):
    def __init__(self, images):
        super().__init__(images)
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.images = [images] * 8 # because shearing has 8 posibilities (4 LEFT and 4 RIGHT)
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
                    
        self.images = combination # add all generated images to the newly created Image object


class HitAndMiss(Image):
    def __init__(self, images):
        super().__init__(images)
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.images = [images] * 2 # because hit and miss has 2 posibilities (10 amd 136 kernel)
        self.operation = 'h'
        self.generate_HitAndMiss()

    def generate_HitAndMiss(self):
        

def save_images(img_dir, des, img):
    rename_image = image_dir.split('/')
    folder = des + '/'
    for element in img: # save all the images
        cv2.imwrite(folder + rename_image[len(rename_image)-1].split('.')[0] + element.get_operation() +'.png', element.get_image())
    
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ WORKING AREA $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if __name__ == '__main__':
##    image_dir = 'Crafty_Girls/alphabets/a.png'
##    destination_folder = 'result'
##    img = get_img_object(image_dir)
##    save_image(image_dir, destination_folder, 'original',img)


    '''
    select type of morphology which can yeild 11 possibilities including kernel
    ** hit and miss has 512 possibilities including original
    '''
##    operation = dictionary(3) 
##    isZero = False
##    for i in range(4):
##        for k in range(4):
##            if (operation == 'gradient' and (i == 1 and k == 1)): # gradiant cannot have kernel (1,1)!!
##                continue
##            if (not isZero):
##                save_image(image_dir, '00' , morphology(img, 0, 0, operation))
##                isZero = True
##            elif(i != 0 and k != 0):
##                save_image(image_dir, str(i) + str(k) , morphology(img, i, k, operation))


    

##    processed_image = Zoom(img,1.5)
##    processed_image = shearing(img)
##    save_image(image_dir, operation, processed_image)


##    kernel_combination = []
##    for i in range(pow(2,9)):
##        kernel_combination.append(bin(i)[2:].zfill(9))
##
##    for index in range(512):
##        kernel_number = index
##        save_image(image_dir, destination_folder, str(kernel_number), hit_and_miss(img, kernel_combination, kernel_number))


##    operation = dictionary(7) 
##    processed_image = shearing(img,True) # True means shearing left
##    save_image(image_dir,operation,processed_image)



##    operation = dictionary(5)
##    processed_image = morphology(img, 1, 2, operation)
##    save_image(image_dir, destination_folder, operation,processed_image)







##    image_dir = 'Crafty_Girls/alphabets/a.png'
##    destination_folder = 'result(1)'
##
##    img = get_img_object(image_dir)
##    processed_image = generate_Erosion(img)
##    save_images(image_dir, destination_folder,processed_image)


#''' THE WORKING CODE (BEGIN) '''
    import os,glob
    folder_path = 'Crafty_Girls/alphabets'
    destination_folder = 'result(1)'
    for filename in glob.glob(os.path.join(folder_path, '*.png')):
        image_dir = filename.replace('\\','/')
        img = cv2.imread(image_dir,cv2.IMREAD_UNCHANGED) 
        b,g,r,a = cv2.split(img)
        image_object = Image(a)

##        do_e = Erosion(image_object)
##        do_de = Erosion(Dilation(image_object).get_image())
##        do_ed = Dilation(Erosion(image_object).get_image())
##        do_zoom = Zoom(image_object)
        do_shearing = Shearing(image_object)

        processed_image = do_shearing.get_image()
        save_images(image_dir, destination_folder,processed_image)

#''' THE WORKING CODE (END) '''

##    dic = {
##        '1':5,
##        '2':5,
##        '3':6,
##        '4':6,
##        '5':2,
##        }
##    s = 0
##    
##    summ = 0
##    stuff = list(range(1,6))
##    print('stuff: ',stuff)
##    for i in range(1,6):
##        perm = combinations(stuff,i)#permutations
##        c = 0
##        s2 = 0
##        for i in list(perm):
##            c += 1
##            print(i)
##            s1 = 1
##            for p in i:
##                s1 *= dic[str(p)]
##            s2 += s1
##        summ += s2
##        s += c
##
##        
##
##    '''
##
##    OPERATION (unit in generated image)
##    (1)Erosion = 5
##    (2)Dilation = 5
##    (3)Shearing(left) = 3
##       Shearing(right) = 3
##    (4)Zoom(in) = 3
##       Zoom(out) = 3
##    (5)Hit and Miss = 2
##    =>Total operation: 5291
##
##    Total training dataset: 5291 * 26 = 137566 
##    
##    '''
##    print(s)
##    print('summ: ',summ)

            
            
    
