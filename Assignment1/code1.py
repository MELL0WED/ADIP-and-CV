from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def contrast_enhancement(V, factor):
    V = np.array(V)  # Convert V to a NumPy array
    return np.clip(V * factor, 0, 1)


def rgb_hsv(r,g,b):
    
    r_normalized = [val/255.0 for val in r]
    g_normalized = [val/255.0 for val in g]
    b_normalized = [val/255.0 for val in b]
    h, s, v = [], [], []
    for i in range(len(r)):
        cmax = max(r_normalized[i], g_normalized[i], b_normalized[i])
        cmin = min(r_normalized[i], g_normalized[i], b_normalized[i])
        delta = cmax - cmin

        # Calculate Hue
        if delta == 0:
            h.append(0)  # undefined, but setting to 0 for simplicity
        elif cmax == r_normalized[i]:
            h.append(60 * (((g_normalized[i] - b_normalized[i]) / delta) % 6))
        elif cmax == g_normalized[i]:
            h.append(60 * (((b_normalized[i] - r_normalized[i]) / delta) + 2))
        elif cmax == b_normalized[i]:
            h.append(60 * (((r_normalized[i] - g_normalized[i]) / delta) + 4))

        # Calculate Saturation
        s.append(0 if cmax == 0 else delta / cmax)
        # Calculate Value
        v.append(cmax)
    return h, s, v
     
def main():

    img = Image.open('contrast.jpg')
    rgb_values = list(img.getdata())
    r,g,b = zip(*rgb_values)
    
    H,S,V = rgb_hsv(r,g,b)

    # Apply contrast enhancement to the Value component
    factor = 1.0  # adjust this value to get the desired level of contrast
    V = contrast_enhancement(V, factor)

    # Reshape H, S, V arrays to the original image shape
    H = np.array(H).reshape(img.size[::-1])
    S = np.array(S).reshape(img.size[::-1])
    V = np.array(V).reshape(img.size[::-1])

    # Stack H, S, V arrays along the third dimension
    hsv_img = np.stack((H, S, V), axis=2)

    # Display the HSV image
    plt.imshow(hsv_img, cmap='hsv')
    plt.show()


if __name__ == "__main__":
    main()