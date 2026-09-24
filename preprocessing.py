import cv2
import numpy as np

def aplica_auto_crop_patrat(img):
    """
    Elimină fundalul negru neproductiv și centrează ochiul într-un pătrat perfect.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w > 20 and h > 20:
            center_x, center_y = x + w // 2, y + h // 2
            side_length = max(w, h)
            half_side = side_length // 2
            x_new, y_new = max(0, center_x - half_side), max(0, center_y - half_side)
            x_end, y_end = min(img.shape[1], center_x + half_side), min(img.shape[0], center_y + half_side)

            img_crop = img[y_new:y_end, x_new:x_end]
            if img_crop.size > 0:
                return img_crop

    return img

def aplica_clahe_lab(img):
    """
    Aplică algoritmul CLAHE exclusiv pe canalul L (Luminozitate) din spațiul LAB.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_eq = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def aplica_ben_graham_verde_x3(img):
    """
    Extrage canalul Verde, aplică normalizarea Ben Graham și îl duplică (x3).
    """
    g = img[:, :, 1]
    sigmaX = img.shape[1] / 30.0
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=sigmaX)
    g_bg = cv2.addWeighted(g, 4, blur, -4, 128)
    return cv2.merge([g_bg, g_bg, g_bg])

def aplica_masca_circulara(img, target_size):
    """
    Aplică masca circulară pentru a elimina artefactele de graniță.
    """
    img_resized = cv2.resize(img, target_size)
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    cv2.circle(mask, (target_size[0]//2, target_size[1]//2), target_size[0]//2, 255, -1)
    return cv2.bitwise_and(img_resized, img_resized, mask=mask)

def shade_of_gray_cc(img, power=6):
    """
    Estimează iluminantul pe baza normei Minkowski.
    """
    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, axis=(0, 1)), 1 / power)
    rgb_vec = rgb_vec / np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = 1.0 / (rgb_vec * np.sqrt(3))
    img = img * rgb_vec.reshape((1, 1, 3))
    return np.clip(img, 0, 255).astype('uint8')