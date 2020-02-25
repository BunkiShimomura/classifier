import cv2

import os
import glob

image_path = input("Enter path to image folder: ")
image_list = sorted(glob.glob(image_path + '/*.jpg'))

#出力先の指定
os.mkdir(image_path + "/output")
output_dir = image_path + "/output"

for i in range(len(image_list)):
    image = cv2.imread(image_list[i])
    file_path = os.path.join(str(output_dir) + "/" + str(i) + "_")
    if image is None:
        pass
    else:
        try:
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(str(file_path) + "_tamago" + ".jpg", image)
        except:
            pass
