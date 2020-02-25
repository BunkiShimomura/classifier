#Reference
#https://qiita.com/wwacky/items/98d8be2844fa1b778323
#https://teratail.com/questions/149285
#https://qiita.com/nirs_kd56/items/bc78bf2c3164a6da1ded

import cv2

import os
import glob

#HAAR分類器の顔検出用の特徴量
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
#cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml"

#カスケード分類機の特徴量を取得
cascade_path = "/Users/Bunki/Desktop/PMP/pytorch_tutorial/script/face/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_path)

#ソース画像の取得
image_path = input("Enter path to image folder: ")
image_list = sorted(glob.glob(image_path + '/*.jpg'))

#出力先の指定
os.mkdir(image_path + "/output")
output_dir = image_path + "/output"

color = (255, 255, 255)

for i in range(len(image_list)):
    image = cv2.imread(image_list[i])
    if image is None:
        pass #NoneTypeが検出すされた時はスキップする
    else:
        image = image.astype('uint8') #https://github.com/llSourcell/Object_Detection_demo_LIVE/issues/6
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
        #顔が複数検出された場合
        #print(len(facerect)) #検出された顔の数
        num = 0
        if len(facerect) > 0:
            #検出した顔を囲む矩形の作成
            file_path = os.path.join(str(output_dir) + "/" + str(i) + "_")
            for rect in facerect:
                img = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                img = cv2.resize(img, (64, 64))
                #認識結果の保存
                cv2.imwrite(str(file_path) + str(num) + "_yoshioka" + ".jpg", img)
                num += 1
        else:
            #print("no face")
            continue
