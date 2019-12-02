import TensorMap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os

np.random.seed(100)

def generate_test_data(frame_num,bone_num,miss_rat=0,inc_rat=0):
    n = frame_num
    m = bone_num
    rbone2gbone = np.zeros( [n,m],np.int )
    gbone2rbone = np.zeros( [n,m],np.int )
    P_list = np.zeros([n,n],np.ndarray)
    mlist = np.ones([n],np.int) * m

    #generate bone for each frame
    for i in range(n):
        # generate mlist[i]
        for j in range(m):
            will_miss = np.random.rand(1)
            if(will_miss < miss_rat):
                mlist[i]-=1
        # those who will show in the frame
        q = np.random.choice(range(m),mlist[i],0)

        # their new index
        rbone2gbone[i][q] = np.random.permutation( range(1,mlist[i]+1) )
        rbone2gbone[i] -=1
        #print('rbone2gbone',rbone2gbone[i])
        for j in range(m):
            if rbone2gbone[i][j] != -1:
                gbone2rbone[i][rbone2gbone[i][j]] = j + 1
        gbone2rbone[i] -=1
        #print('rbone2gbone', rbone2gbone[i])

    # generate Plist
    for i in range(n):
        for j in range(n):
            res = np.zeros([mlist[j],mlist[i]],np.int)
            for k in range(mlist[j]):
                rindex = gbone2rbone[j][k]
                pindex = rbone2gbone[i][rindex]
                if pindex !=-1:
                    res[k][pindex] = 1

            will_inc = np.random.rand(1)
            if will_inc < inc_rat:
                x = np.random.randint(0,mlist[j])
                y = np.random.randint(0,mlist[j])
                res[x,:],res[y,:] = res[y,:],res[x,:]
            P_list[i][j] = res.copy()

    return P_list,mlist,rbone2gbone,gbone2rbone


def generate_image(num):
    img = np.zeros([256,256],np.int)
    lastline = 0
    t = int(256/num)
    for i in range(num):
        img[lastline:lastline+t,:] = i
        lastline+=t
    img[lastline:256,:] = num-1

    return img

n = 20
m = 5
mis = 0.1
inc_rat = 0.2
curpath = 'D:\\paper\\testtensor'

data = generate_test_data(n,m,mis,inc_rat)
Plist, mlist, rbone2gbone, gbone2rbone = data


tensor = TensorMap.SynTensorMap(n,mlist,Plist)
Q = tensor.solution()