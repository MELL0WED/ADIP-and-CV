import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def BGR_HSV_CONVERSION(b, g, r):
    b, g, r = b/255.0, g/255.0, r/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def HSV_BGR_CONVERSION(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
    return int(b), int(g), int(r)

def normalize_rgb(rgb):
    rgb_sum = np.sum(rgb, axis=-1, keepdims=True)
    return rgb / rgb_sum

def Chromaticity_calculation(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_norm = normalize_rgb(img_rgb)
    return img_rgb_norm[..., 0].ravel(), img_rgb_norm[..., 1].ravel()

def main():
    img = cv2.imread('flower.jpg')
    h, w, _ = img.shape
    hsv_img = np.zeros((h, w, 3))
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            h, s, v = BGR_HSV_CONVERSION(b, g, r)
            s = 1.0
            b, g, r = HSV_BGR_CONVERSION(h, s, v)
            hsv_img[i, j] = h, s, v
            rgb_img[i, j] = b, g, r

    cv2.imwrite('saturated_flower.jpg', rgb_img)

    img = cv2.imread('flower.jpg')
    h, w, _ = img.shape
    hsv_img = np.zeros((h, w, 3))
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            h, s, v = BGR_HSV_CONVERSION(b, g, r)
            s = 0.0
            b, g, r = HSV_BGR_CONVERSION(h, s, v)
            hsv_img[i, j] = h, s, v
            rgb_img[i, j] = b, g, r

    cv2.imwrite('flower_desaturated.jpg', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    img = cv2.imread('flower.jpg')
    h, w, _ = img.shape
    hsv_img = np.zeros((h, w, 3))
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            h, s, v = BGR_HSV_CONVERSION(b, g, r)
            s = 1.0
            b, g, r = HSV_BGR_CONVERSION(h, s, v)
            hsv_img[i, j] = h, s, v
            rgb_img[i, j] = b, g, r
            b, g, r = img[i, j]
            h, s, v = BGR_HSV_CONVERSION(b, g, r)
            s = 0.0
            b, g, r = HSV_BGR_CONVERSION(h, s, v)
            hsv_img[i, j] = h, s, v
            rgb_img[i, j] = b, g, r

    cv2.imwrite('flower_sat_desat.jpg', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))


    # Display the original, saturated, and desaturated images in the same plot
    fig, axs = plt.subplots(1, 4, figsize=(24, 8))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')

    axs[1].imshow(cv2.cvtColor(cv2.imread('saturated_flower.jpg'), cv2.COLOR_BGR2RGB))
    axs[1].set_title('Saturated Image')

    axs[2].imshow(cv2.cvtColor(cv2.imread('flower_desaturated.jpg'), cv2.COLOR_BGR2RGB))
    axs[2].set_title('Desaturated Image')

    axs[3].imshow(cv2.cvtColor(cv2.imread('flower_sat_desat.jpg'), cv2.COLOR_BGR2RGB))
    axs[3].set_title('SatDesat Image')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


    # Load the data
    column_names = ['Wavelength (in nm)', 'X', 'Y', 'Z']
    data = pd.read_csv('ciexyz31_1.csv', names=column_names)

    # Calculate the chromaticity coordinates
    data['x'] = data['X'] / (data['X'] + data['Y'] + data['Z'])
    data['y'] = data['Y'] / (data['X'] + data['Y'] + data['Z'])

    img_original = cv2.imread('flower.jpg')
    img_saturated = cv2.imread('saturated_flower.jpg')
    img_desaturated = cv2.imread('flower_desaturated.jpg')
    img_satdesat = cv2.imread('flower_sat_desat.jpg')

    # Calculate chromaticity for each image
    x_original, y_original = Chromaticity_calculation(img_original)
    x_saturated, y_saturated = Chromaticity_calculation(img_saturated)
    x_desaturated, y_desaturated = Chromaticity_calculation(img_desaturated)
    x_satdesat, y_satdesat = Chromaticity_calculation(img_satdesat)

    fig, axs = plt.subplots(4, 1, figsize=(8, 24))

    # Plot the chromaticity points for the original image
    axs[0].scatter(x_original, y_original, alpha=0.1, s=1)
    axs[0].set_title('Chromaticity Points of the Original Image')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])

    # Add a bit of gap between caption and the next image
    axs[0].title.set_position([0.5, 1.05])

    # Plot the chromaticity points for the saturated image
    axs[1].scatter(x_saturated, y_saturated, alpha=0.1, s=1)
    axs[1].set_title('Chromaticity Points of the Saturated Image')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    # Add a bit of gap between caption and the next image
    axs[1].title.set_position([0.5, 1.05])

    # Plot the chromaticity points for the desaturated image
    axs[2].scatter(x_desaturated, y_desaturated, alpha=0.1, s=1)
    axs[2].set_title('Chromaticity Points of the Desaturated Image')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_xlim([0, 1])
    axs[2].set_ylim([0, 1])

    # Add a bit of gap between caption and the next image
    axs[2].title.set_position([0.5, 1.05])

    # Plot the chromaticity points for the satdesat image
    axs[3].scatter(x_satdesat, y_satdesat, alpha=0.1, s=1)
    axs[3].set_title('Chromaticity Points of the Desaturated Image')
    axs[3].set_xlabel('x')
    axs[3].set_ylabel('y')
    axs[3].set_xlim([0, 1])
    axs[3].set_ylim([0, 1])

    # Add a bit of gap between caption and the next image
    axs[3].title.set_position([0.5, 1.05])

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(data['x'], data['y'], c=data['Wavelength (in nm)'], cmap='jet', alpha=0.5)
    plt.title('Chromaticity Points of the Color Matching Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.colorbar(label='Wavelength (in nm)')
    plt.show()
    
    

if __name__ == "__main__":
    main()
