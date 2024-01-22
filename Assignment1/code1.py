from PIL import Image
import numpy as np

def rgb_hsv(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

def main():
    #img = np.array(Image.open('contrast.jpg'))
    img = Image.open('contrast.jpg')
    rgb_values = list(img.getdata())
    #print(len(rgb_value))
    width, height = img.size
    rgb_values = [rgb_values[i:i+width] for i in range(0, len(rgb_values), width)]
    #print(np.shape(rgb_values))
    hsv_img = rgb_hsv(rgb_values)

if __name__ == "__main__":
    main()