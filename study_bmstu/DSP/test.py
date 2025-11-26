from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('C:/Users\superpro2005\Desktop\study\python\photo\car.jpg')
model = LatexOCR()
print(model(img))