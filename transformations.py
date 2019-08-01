import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
#from neupeak.utils import webcv2 as cv2
import numpy as np
from scipy.io import loadmat
#from neupeak.utils.imgproc import montage
import skimage

def ShearX(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v*img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):  # [-30, 30]
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

def Solarize(img, v):  # [0, 256]
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):  # [4, 8]
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    w, h = img.size
    v = v*img.size[0]
    x0 = np.random.uniform(w-v)
    y0 = np.random.uniform(h-v)
    xy = (x0, y0, x0+v, y0+v)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Gaussion(img, v):  # [0, 20]
    img = np.asarray(img) / 255.
    img = skimage.util.random_noise(img, var = v)
    img *= 255.
    img = PIL.Image.fromarray(img.astype('uint8'))
    return img

def get_transformations(imgs):
    return [
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (TranslateX, -0.45, 0.45),
        (TranslateY, -0.45, 0.45),
        (Rotate, -30, 30),
        (AutoContrast, 0, 1),
        (Invert, 0, 1),
        (Equalize, 0, 1),
        (Solarize, 0, 256),
        (Posterize, 4, 8),
        (Contrast, 0.1, 1.9),
        (Color, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (Cutout, 0, 0.2),
        (Gaussion, 0, 0.1),
    ]

if __name__ == '__main__':
    imgs = 1 
    transfs = get_transformations(imgs)
    for i in range(10):
        for t, min, max in transfs:
            while True:
                img2 = img = PIL.Image.open('test.jpg')
                v = np.random.rand()*(max-min) + min
                img2 = t(img2, v)
                board = montage([np.asarray(img), np.asarray(img2)], 1)
                cv2.imshow('', board)
                c = chr(cv2.waitKey(0) & 0xff)
                if c == 'q':
                    exit()
 
