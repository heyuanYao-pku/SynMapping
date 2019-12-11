import json
import cv2
import os
import numpy as np
import TensorMap
data_path = 'example\\data.json'
data = json.load(open( data_path,'r') )
print(data.keys())
mList = data['mlist']
tmp = data['P']
image_list_path = 'example\\image_list.npy'
image_list = np.load(image_list_path)

#mList = mList[0:20]
#tmp = np.array(tmp)
#tmp = tmp[0:20,0:20]
#image_list = image_list[0:20]

#print(  np.shape(Plist)  )
n = np.size(mList)
Plist = np.zeros(np.shape(tmp), np.ndarray)

print(tmp[13][14])
#exit(0)
print(  n  )
#print(mList[0],mList[100],mList[2],mList[102] )


near = 10
for i in range(n):
    for j in range(n):
        if i==j:
            Plist[i][j] = np.eye(mList[i])
            continue
        '''
        if abs(i-j) > near and not( i < near and j < 2*near)  \
                and not( (n-i) < near and (n-j) < 2*near) :
            Plist[i][j] = np.zeros(np.shape(tmp[j][i]),np.float)
            continue
        '''
        Plist[i][j] = np.array( tmp[j][i])


tensor = TensorMap.SynTensorMap(n,mList,Plist)
#Q = tensor.rounded_solution(th=0.5,k=1)
Q = tensor.multi_round(th = 0.5)
#Q = np.load('example\\Q.npy')


#print(mList[0:2],Q)
for i in range(len(Q)):
    print(i, Q[i])


draw = TensorMap.TensorMapVis(image_list,mList,Q).draw_image

save_path = 'example\\'
for i in range(n):
    cv2.imwrite(os.path.join(save_path,'%d.png'%i),draw[i])
