import numpy as np
import os
import json
from Accuracy import colorspace as cs
from Accuracy import AccurCal as AC
import cv2
#  给你发的文件夹  每个是一组视频，每张图片有个同名的json
# json里面有个paf
# a = json.load(xxxx)
# points = a['paf']['paf']
# class_index =a['paf']['class_index']

def get_bone_and_seg(keypoints, class_index,sigma_paf=8):
     # 输入的一些点对的list，list长度是骨骼数目，每个骨骼有两个点
     # class index的长度是 和keypoints的长度一样的，代表骨骼的标签
    this_size = 256
    w, h = this_size, this_size
    out_pafs_this = np.zeros((1, this_size, this_size))

    mask_use = np.ones((this_size,this_size),dtype=bool)
   
    truth_index =0
    for paf_index in range (len(keypoints)-1,-1,-1):
        class_value = class_index[paf_index]
        keypari  = keypoints[paf_index]
        keypoint_1 = np.array(keypari[0])/2
        keypoint_2 = np.array(keypari[1])/2
        part_line_segment = keypoint_2 - keypoint_1
        l = np.linalg.norm(part_line_segment)
        if l > 1e-2:
            sigma = sigma_paf
            # 这个地方是骨骼的宽度
            v = part_line_segment / l
            v_per = v[1], -v[0]
            x, y = np.meshgrid(np.arange(h), np.arange(w))
            dist_along_part = v[0] * (x - keypoint_1[0]) + v[1] * (y - keypoint_1[1])
            dist_per_part = np.abs(v_per[0] * (x - keypoint_1[0]) + v_per[1] * (y - keypoint_1[1]))
            #这里用来画这个宽度的骨骼
            mask1 = dist_along_part >= 0
            mask2 = dist_along_part <= l
            mask3 = dist_per_part <= sigma
            mask = mask1 & mask2 & mask3 & mask_use
            out_pafs_this[0][mask > 0] = class_value

            mask_use[mask>0] = False
            # 用来表示表示遮挡，先画上的就把后面的挡住，挡住，不许画

    return out_pafs_this



if __name__ == '__main__':
    path1 = r'C:\Users\16000\Desktop\Research\TensorMap\gqzdata1227data\5_C_Swim2'
    path2 = r'C:\Users\16000\Desktop\Research\TensorMap\swim2\finall_res\epoch175'
    '''
    curpic = os.path.join(path1,'5_C_Swim2_0_3.json')
    f = open(curpic,'r')
    x = json.load(f)
    img = get_bone_and_seg(x['paf']['paf'],x['paf']['class_index'])
    img2 = np.zeros((256,256,3),np.float)
    cs2 = cs()
    for i in range(256):
        for j in range(256):
            img2[i][j] = cs2[ int(img[0][i][j]) ]
    print(np.unique(img),np.unique(x['paf']['class_index']))
    cv2.imshow("sss",img2)
    cv2.waitKey()
    '''
    score = 0
    calculator = AC((256, 256))
    for i in range(2,53):
        curpic = os.path.join(path1,'5_C_Swim2_0_%d.json'%i)
        f = open(curpic, 'r')
        x = json.load(f)
        img = get_bone_and_seg(x['paf']['paf'], x['paf']['class_index'])
        img = img.reshape((256,256))
        img = np.array(img,np.int)
        print(img.shape)
        curdet = os.path.join(path2,'%04d.npy'%i)
        det = np.load(curdet)
        det = np.array(det, np.int)
        #print(det)
        s,m = calculator.calculate_one_frame(img,det,visualize=False)
        print('frame %d'%i, s, m)
        score +=s/m
    print(score)

