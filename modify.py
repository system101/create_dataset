from PIL import ImageDraw, ImageFont
from PIL import Image as I
import numpy as np
import matplotlib.pyplot as plt
import cv2

def text_phantom(font_dir, text, size):
    # Availability is platform dependent
    font = font_dir

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)
    print('\ntext_width: ',text_width)
    print('text_height: ',text_height)

    # create a blank canvas with extra space between lines
    canvas = I.new('L', [size, size], (255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              ((size - text_height) // 2))
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white,align="center")

    # Convert the canvas into an array with values in [0, 255] black background
    return 255-np.asarray(canvas)

def get_alphabets(file_dir):
  alphabets = []
##  for i in range(97,123): # 'a' to 'z'
  for i in range(65,91): # 'A' to 'Z'
    alphabets.append(text_phantom(file_dir,chr(i), 28))
  return np.asarray(alphabets)


font = 'black-and-bitter-coffee/Black-and-Bitter-Coffee-TTF'
alphabets = get_alphabets(font)
##for m in alphabets:
##    plt.imshow(m,cmap='gray')
##    plt.show()



def generate_Shearing(alphabets):
    '''
    shear left range [0.6,0.9]
    shear right range [0.1,0.4]
    '''
    combination = []
    leftRight = [0,1]
##        left_and_right_shear = [[0.1,0.2,0.3,0.4],[0.6,0.7,0.8,0.9]] # right and then left
    left_and_right_shear = [[0.1,0.2],[0.8,0.9]] # right and then left

    for isShearingLeft in leftRight: #[0,1]
        for angle in left_and_right_shear[isShearingLeft]:
            img = alphabets[24]
            y, x = img.shape[:2]
            distance_from_origin = angle
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
            plt.imshow(img_output,cmap='gray')
            plt.show()



generate_Shearing(alphabets)
