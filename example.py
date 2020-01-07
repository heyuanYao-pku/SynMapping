import TensorMap
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import cluster_image
np.random.seed(88)

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
            if(will_miss < miss_rat and mlist[i]!=1):
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

def draw_input_data(n,m,mlist,gbone2rbone,inputpath):
    color_num = m
    cmap = plt.get_cmap('gist_ncar')
    colors_set = cmap(np.linspace(0, 1, color_num + 1))[:, 0:3]
    img = generate_image(m)
    cont = 0
    for i in range(n):
        fimg = np.zeros([256, 256, 3], np.float32)
        tmp = mlist[i]
        for j in range(cont, cont + mlist[i]):
            ind = gbone2rbone[i][j - cont]
            #print('ind:', ind)
            S = np.where(img == ind)
            l = np.size(S[0])
            # print('k:',k)
            #p = np.array(Q[j])
            t = ind
            # print('P j cont', p, j, cont, mlist[i])
            # print('t', t)
            # print('np:', np.shape( colors_set[t][0:3] * 255) )
            #print(np.shape(colors_set))
            #print(t)
            for k in range(l):
                fimg[S[0][k]][S[1][k]] = colors_set[t] * 255
        cont += mlist[i]
        # print(fimg)
        # cv2.imshow('s',fimg.astype(np.uint8))
        # cv2.imshow('ss',img*255)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(inputpath, '%04d.png' % i), fimg.astype(np.uint8))

def mmkdir(file):
    if os.path.exists(file):
        return
    os.mkdir(file)

n = 40
m = 6
mis = 0.3
inc_rat = 0.5
curpath = 'TmpData/'
inputpath = 'TmpData/input'
mmkdir(inputpath)

data = generate_test_data(n,m,mis,inc_rat)
Plist, mlist, rbone2gbone, gbone2rbone = data
draw_input_data(n,m,mlist,gbone2rbone,inputpath)

labels  = cluster_image.cluster_image(inputpath,4 )
#np.save(os.path.join(curpath,'data.npy'),data)
near = 7

check = np.zeros((n,n),np.int)

extra_edge_num = 0
edge_num = 0
for i in range(n):
    for j in range(n):
        #if abs(i-j) > near and not (i<near and j <2*near) and not (j<near and i<2*near) \
        #    and not ((n-i) <near and (n-j) < 2*near) and not ((n-j)<near and (n-i) < 2*near) :
        if abs(i-j) > near and not (i<near and j <2*near)  \
            and not ((n-i) <near and (n-j) < 2*near) and not (labels[i] == labels[j]):
            Plist[i][j] = np.zeros( np.shape(Plist[i][j]) , np.float)
            continue;
        check[i][j] = 1
        edge_num +=1
        if abs(i - j) > near and not (i < near and j < 2 * near) \
                and not ((n - i) < near and (n - j) < 2 * near) and  (labels[i] == labels[j]):
            check[i][j]+=1
            extra_edge_num +=1
print(check)
print("extra_num", extra_edge_num, "edge num", edge_num)
tensor = TensorMap.SynTensorMap(n,mlist,Plist)
Q = tensor.rounded_solution(0.5)
#Q = np.load(os.path.join(curpath,'sol.npy'))
img = generate_image(m)
#np.save(os.path.join(curpath,'sol.npy'),Q)

color_num = np.size(Q[0])
cmap = plt.get_cmap('gist_ncar')
colors_set = cmap(np.linspace(0, 1,color_num+1))[:,0:3]


cont = 0
for i in range(n):
    fimg = np.zeros([256,256,3],np.float32)
    tmp = mlist[i]
    for j in range(cont,cont+mlist[i]):
        ind = gbone2rbone[i][j-cont]
        #print('ind:',ind)
        S = np.where(img==ind)
        l = np.size(S[0])
        # print('k:',k)
        p = np.array(Q[j])
        t = np.where(p == 1)[0][0]
        #print('P j cont', p, j, cont, mlist[i])
        #print('t', t)
        # print('np:', np.shape( colors_set[t][0:3] * 255) )

        for k in range(l):
            fimg[ S[0][k] ][ S[1][k] ] = colors_set[t] * 255
    cont += mlist[i]
    #print(fimg)
    #cv2.imshow('s',fimg.astype(np.uint8))
    #cv2.imshow('ss',img*255)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(curpath,'%04d.png'%i),fimg.astype(np.uint8))