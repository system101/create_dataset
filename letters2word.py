from PIL import Image, ImageFont, ImageDraw


w, h = 28 , 28
ttf_file_dir = 'Sunshiney/Sunshiney.ttf'
# use a truetype font
font = ImageFont.truetype(ttf_file_dir,28)
##im = Image.new("RGBA", (28, 28))
##draw = ImageDraw.Draw(im)

for code in range(ord('y'), ord('y') + 1):
##  w, h = draw.textsize(chr(code), font=font)
##  print('w: ', w)
##  print('h: ', h)
  im = Image.new("RGBA", (w, h))
  draw = ImageDraw.Draw(im)
  draw.text((5, -10), chr(code), font=font, fill="#000000")
  im.save(chr(code) + ".png")
