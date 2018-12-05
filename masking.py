import numpy as np
import cv2 as cv

in_path = '/mnt/nas/data/selfassembly/cam1/'
out_path = './output/'

# read background image
bg_img = cv.imread('default.jpg', cv.IMREAD_GRAYSCALE)
blur_bg = cv.GaussianBlur(bg_img, (5,5), 0) / 255

def make_mask(path):
  # read target image
  tg_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
  blur_tg = cv.GaussianBlur(tg_img, (5,5), 0) / 255
  # find mask for objects
  diff = cv.pow(abs(blur_bg - blur_tg), 2)
  return (diff>0.3).astype(np.uint8)

for i in range(10000):
  in_file = '%07d' % (i+10)
  mask = make_mask(in_path + in_file + '.jpg')
  org_img = cv.imread(in_path + in_file + '.jpg')
  org_img[mask>0] = [200,0,200]
  cv.imwrite(out_path + in_file + '.png', org_img)

