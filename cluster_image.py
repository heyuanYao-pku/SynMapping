import cv2
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import os

def walk_dir(file):
    ''':return  root, dirs, files'''
    for root, dirs,files in os.walk(file):
        return root, dirs, files
        pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cluster_image(dir_path,cluster_num):
    _,_,files_name_list = walk_dir(dir_path)
    all_image_feature  = []
    #files_name_list = files_name_list[4:72]
    print(files_name_list)
    ###  对原图或者bone进行cluster，具体哪个效果好不好说
    for index, file_name in enumerate(files_name_list):

        #img = cv2.imread(dir_path+'/'+file_name)
        img = np.load(dir_path+'/'+file_name)
        img[img!=0]/=img[img!=0]/8

        imag_array = img.flatten()
        # print(gray.dtype)
        all_image_feature.append(imag_array)

    all_image_feature = np.array(all_image_feature)

    # pca 分解的 维度可以自己设置
    pca = sklearn.decomposition.PCA(n_components=30)

    input_x =  pca.fit_transform(all_image_feature)
    # 这里你可以换别的聚类的办法
    kmeans = KMeans(n_clusters=cluster_num).fit(input_x)

    # 这样每张图片按顺序都获得一个类的label 
    cluster_labels =  kmeans.labels_

    return  cluster_labels

#path = 'human_hand_more_seg\\bone_source_data'
#x = cluster_image(path,5)
