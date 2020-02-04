##from PIL import Image, ImageFont, ImageDraw
##
##
##w, h = 28 , 28
##ttf_file_dir = 'Sunshiney/Sunshiney.ttf'
### use a truetype font
##font = ImageFont.truetype(ttf_file_dir,28)
####im = Image.new("RGBA", (28, 28))
####draw = ImageDraw.Draw(im)
##
##for code in range(ord('y'), ord('y') + 1):
####  w, h = draw.textsize(chr(code), font=font)
####  print('w: ', w)
####  print('h: ', h)
##  im = Image.new("RGBA", (w, h))
##  draw = ImageDraw.Draw(im)
##  draw.text((5, -10), chr(code), font=font, fill="#000000")
##  im.save(chr(code) + ".png")



from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

def text_phantom(font_dir, text, size):
    # Availability is platform dependent
    font = font_dir

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('L', [size, size], (255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)
    plt.imshow(draw)
    # Convert the canvas into an array with values in [0, 255] black background
    return 255-np.asarray(canvas)

def get_alphabets(file_dir):
  alphabets = []
  for i in range(97,123): # 'a' to 'z'
    alphabets.append(text_phantom(file_dir,chr(i),64))
  return np.asarray(alphabets)


font = 'black-and-bitter-coffee/Black-and-Bitter-Coffee-TTF'
alphabets = get_alphabets(font)
t = alphabets[25]
print(t.shape)
plt.imshow(t,cmap='gray')
plt.show()
