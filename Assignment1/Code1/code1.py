import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def BRG_RGB_CONVERSION(b, g, r):
    return r, g, b

def contrast_enhancement(V, factor):
    V = np.array(V)  # Convert V to a NumPy array
    return np.clip(V * factor, 0, 1)

def main():
    img = cv2.imread('contrast.jpg')
    h, w, _ = img.shape
    hsv_img = np.zeros((h, w, 3))
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            h, s, v = BGR_HSV_CONVERSION(*img[i, j])
            if s < 0.1:  # Check if the pixel is achromatic
                v = contrast_enhancement(v, 1.3)  # Apply contrast enhancement to V
            hsv_img[i, j] = [h, s, v]
            rgb_img[i, j] = BRG_RGB_CONVERSION(*HSV_BGR_CONVERSION(*hsv_img[i, j]))
    plt.imshow(rgb_img)
    plt.show()

    # Save the enhanced image
    cv2.imwrite('enhanced_image.jpg', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
