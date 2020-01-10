import numpy as np
import cv2
import matplotlib.pyplot as plt

class colorspace:
    def __init__(self):
        self.color = [(255,255,255), (240, 255, 240), (137, 104, 205), (0, 191, 255),(255,130,71),
                      (255, 165, 0), (47, 79, 79), (110,139,61) , (255, 62, 150),
                      (255, 255, 0), (32, 178, 170), (255, 222, 173), (205, 92, 92),
                      (178, 58, 238), (70, 130, 180), (255, 20, 147), (230, 230, 250),
                       (0, 238, 238), (173, 255, 47), (0, 0, 255), (250, 128, 114),

                      ]
        self.color = np.array(self.color)/255
        self.n = len(self.color)

    def visualize(self):

        width = 15
        img = np.zeros( (self.n*width,256,3) ,np.float )

        for cnt in range(self.n):
            for i in range(cnt*width, cnt*width+width):
                for j in range(256):
                    img[i][j] = self[cnt]
        cv2.imshow("colorspace",img)
        cv2.waitKey()

    def __getitem__(self, item):
        return self.color[item]


class AccurCal:
    def __init__(self, shape):
        self.shape = shape
        self.maxlabel = 0
        self.mapping = {}
        self.cur_frame = None
        self.cur_detect = None

    def comparing(self, frai, detj, returnextra = False):
        indi = np.where(self.cur_frame == frai)
        indj = np.where(self.cur_detect == detj)

        indi = np.transpose(indi)
        indj = np.transpose(indj)
        indi = {tuple(x) for x in indi}
        indj = {tuple(x) for x in indj}

        intersection = indi & indj
        unionset = indi | indj

        if len(indi) == 0:
            rate = 0
        else:
            rate = len(intersection)/len(indi)
        if returnextra:
            return rate, len(intersection), len(unionset), intersection, unionset
        #print(frai, detj, rate, len(indj), len(intersection), len(unionset) )
        return rate, len(intersection), len(unionset)

    def calculate_one_frame(self, frame, detection, visualize = True, change_map = True):

        assert frame.shape == self.shape,"frame shape not equal to shape given"
        assert detection.shape == self.shape, "detection shape not equal to shape given"
        self.cur_detect = detection
        self.cur_frame = frame

        l_d = np.min(detection[detection>1])
        u_d = np.max(detection)

        l_f = np.min(frame)
        u_f = np.max(frame)

        map_success = np.zeros(u_d+1, np.int)
        score = 0
        part_num = 0
        for i in range(l_d,u_d+1):

            rate,inter,unio = None, None, None

            if i not in self.mapping:

                if not change_map:
                    continue

                for j in range(l_f, u_f+1):

                    if j in self.mapping.values():
                        continue
                    rate, inter, unio = self.comparing(j,i)

                    if rate >=0.5:
                        self.mapping[i] = j
                        break
                else:
                    continue
            else:
                rate, inter, unio = self.comparing( self.mapping[i],i)

            print("final rate for detect",i,"is",rate)
            if rate >=0.5:
                score += 1
                map_success[i] = 1
            if i in detection:
                part_num += 1
        if visualize:
            print("map is",self.mapping)
            self.visualize(frame, detection, map_success)

        return score,part_num

    def calculate_video(self, GTvideo, detection):

        score = 0
        nframes = 0

        for f,d in zip(GTvideo,detection):
            s,p = self.calculate_one_frame(f,d)
            score+=s/p

        return score/nframes

    def visualize(self, frame, detect, map_succ):
        cs = colorspace()
        plt.figure("GT DET")
        #print(map_succ)
        #print(self.mapping)
        #cs.visualize()
        l_d = np.min(detect)
        u_d = np.max(detect)

        img = np.zeros( (self.shape[0],self.shape[1]*2,3) , np.float)


        w,h = self.shape

        for i in range(w):
            for j in range(h):
                img[i][j] = cs[frame[i][j]]

        q = np.unique(frame)

        center = []
        for i in q:
            cloud = np.array(np.where(frame == i))
            cloud = np.transpose(cloud)
            center.append((np.mean(cloud, axis=0), "%d" % i))


        for j in range(l_d,u_d+1):
            if not map_succ[j]:

                cloud = np.array(np.where(detect == j))

                cloud[1] += self.shape[1]  # np.array(cloud[0]) + self.shape[0] \
                cloud = tuple(cloud)
                # print(np.unique(cloud))
                img[cloud] = (1,0,0) if j!=1 else cs[0]
                continue

            i = self.mapping[j]

            '''
            cloudi = np.where(detect == i)
            cloudj = np.where(frame == j)
            cloudi, cloudj = cloudi.transpose(), cloudj.transpose()
            centeri, centerj = np.mean(cloudi,axis= 0), np.mean(cloudj,axis= 0)
            centerj[1] += self.shape[1]
            '''
            cloud = np.array( np.where(detect == j) )

            cloud[1] += self.shape[1] #np.array(cloud[0]) + self.shape[0] \
            cloud = tuple(cloud)
            #print(np.unique(cloud))
            img[cloud] = cs[i]
            cloud = np.transpose(cloud)
            center.append( (np.mean(cloud,axis=0) , "%d"%i ) )


        plt.imshow(img)
        for c,txt in center:

            x,y = c
            #print(int(x),int(y),txt)
            plt.text(int(y),int(x),txt,fontdict={'fontsize':12})
        #plt.axes('on')
        plt.show()

if __name__ == '__main__':
    c = colorspace()
    c.visualize()
