import numpy as np
import sys
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Array
import cv2
import time
import os
from multiprocessing import Process, Manager

class SynTensorMap:
    def __init__(self, shape_num, part_num_list, Map_list, Param_str_list = [], Not_Build = False):
        '''
        :param shape_num: The num of all shapes
        :param part_num_list: Nums of parts for each shape
        :param Map_list: The Mapping matrix for each (i,j),
                where Pij is the map from shape j to i and it's shape is mj * mi
        :param Not_Build: Set true to avoid redundant building
        :param Iter_tol: Iteration tolerance
        '''

        self.n = shape_num
        self.mList = part_num_list.copy()
        self.Plist = Map_list.copy()
        self.N = sum(part_num_list)  # Total number of all parts
        self.indBegin, self.indEnd = self.GetInd() # Where shape i begin and end

        self.set_param(Param_str_list)

        if self.paramkey['SAVE_TMP']:
            if not os.path.isdir(self.paramkey['SAVE_PATH']):
                os.mkdir(self.paramkey['SAVE_PATH'])

        np.random.seed(99)


        if Not_Build==False: # Check Inputs

            assert np.size(part_num_list) == shape_num, 'Length of part num list is not equal to shape num\n'
            assert np.shape(Map_list) == (shape_num,shape_num), 'Shape of Map_list is not equal to (%d, %d)\n' %(shape_num,shape_num)
            for i in range(shape_num):
                for j in range(shape_num):
                    assert np.shape(Map_list[i][j]) == (part_num_list[j],part_num_list[i]) , \
                        'Map_list[%d][%d] is %d * %d, but part num[j]  is %d and part_num[i] is %d\n'\
                        %(i,j,np.shape(Map_list[i][j])[0],np.shape(Map_list[i][j])[1],part_num_list[j],part_num_list[i])

        else:
            return

        ######### Begin Building ########

        print('building P')
        self.buildP()
        print('building C')
        self.buildC()
        print('building R')
        self.buildR()

        self.sol = None
        self.rounded_sol = None

    def GetInd(self):
        '''
        :return: The begin and end index of each part
        '''
        indBegin = self.mList.copy()
        s = 0
        for i in range(self.n):
            indBegin[i] = 0 if i == 0 else indBegin[i - 1] + self.mList[i - 1]
        indEnd = [indBegin[i + 1] if i != self.n - 1 else self.N for i in range(self.n)]
        return indBegin,indEnd

    def buildP(self):
        self.P = np.zeros([self.N,self.N],np.double)
        n = self.n
        for i in range(n):
            for j in range(i,n):
                k = ( sum(sum(self.Plist[i][j])) >= sum(sum(self.Plist[j][i])))
                if k==True:
                    self.P[self.indBegin[j]:self.indEnd[j],self.indBegin[i]:self.indEnd[i]] = self.Plist[i][j]
                    self.P[self.indBegin[i]:self.indEnd[i], self.indBegin[j]:self.indEnd[j]] = np.transpose( self.Plist[i][j] )
                else:
                    self.P[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i]] = np.transpose( self.Plist[j][i] )
                    self.P[self.indBegin[i]:self.indEnd[i], self.indBegin[j]:self.indEnd[j]] = self.Plist[j][i]

    def buildC_func(self,x):
        i, j, k = x
        tmp = np.dot(self.Plist[j][k], self.Plist[i][j])
        tmp = np.dot(self.Plist[k][i], tmp)
        d = np.diag(tmp)
        d = np.diag(d)
        return np.where(d > 0, 1, 0)

    def buildC(self):
        '''
        计算 Cijk =  Pki * Pjk * Pij 并舍弃对角线以外的值
        '''
        n = self.n
        self.Clist = np.zeros([n,n,n],np.ndarray)

        tmp = np.zeros([n,n,n],np.ndarray)


        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tmp = np.dot(self.Plist[j][k],self.Plist[i][j])
                    tmp = np.dot(self.Plist[k][i],tmp)
                    d = np.diag(tmp)
                    d = np.diag(d)
                    self.Clist[i][j][k] = np.where(d > 0, 1,0)


    def getRijk(self,i,j,k):
        '''
        :param i:
        :param j:
        :param k:
        :return: 计算Rijk = P[i][j] * C[i][j][k]
        '''
        return np.dot(self.Plist[i][j],self.Clist[i][j][k])

    def buildR(self):
        '''
        :return:
        Rlist 是分块的tensor
        R是整个的tensor
        '''
        n = self.n
        self.Rlist = np.zeros([n,n,n],np.ndarray)
        self.R = np.zeros([self.N, self.N, n], np.double)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    self.Rlist[i][j][k] = self.getRijk(i,j,k)
                    self.R[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i],k] = self.Rlist[i][j][k]

    def build_Wrst(self):

        '''
        获取R中每个点都和哪个C相乘，方法是先把他们都设成全为1的数组，模拟一次与C相乘的结果就是与之相乘的Cijk
        :return:
        '''
        self.Wrst = np.ones([self.N,self.N,self.n],np.double)
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n): # 取出来一小片
                    tmp = self.Wrst[ self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i], k]
                    tmp = np.dot(tmp,self.Clist[i][j][k])
                    self.Wrst[self.indBegin[j]:self.indEnd[j], self.indBegin[i]:self.indEnd[i], k] = self.Rlist[i][j][k] = tmp


    def solution(self,m_bar_advice = 0):

        self.build_Wrst()

        self.m_bar,self.Ax,self.Bx,self.Cx = self.get_init(m_bar_advice)
        print('m_bar = ',self.m_bar)
        self.Ax = np.array(self.Ax, np.double)
        self.Bx = np.array(self.Bx, np.double)
        self.Cx = np.array(self.Cx, np.double)
        self.e1 = self.e2 = self.e3 = self.paramkey['ITER_TOL']*100;

        cont  = 0
        self.dist = self.paramkey['ITER_DISFUNC']


        while(True):

            tmp1, tmp2,tmp3 = self.Ax.copy(),self.Bx.copy(),self.Cx.copy()

            cont +=1
            print("iter%d"%cont)

            if cont == self.paramkey['ITER_MAXNUM']:
                break

            if(self.e1 > self.paramkey['ITER_TOL']/10) and np.random.randint(1,4)==1:
                self.Ax = self.optA()
                self.e1 = self.dist(tmp1,self.Ax)
            else:
                print("A OK and e1 is ",self.e1)

            if (self.e2 > self.paramkey['ITER_TOL']/10) and np.random.randint(1,4)==1:
                self.Bx = self.optB()
                self.e2 = self.dist(tmp2, self.Bx)
            else:
                print("B OK and e2 is ", self.e2)

            if (self.e3 > self.paramkey['ITER_TOL']/10) and np.random.randint(1,4)==1:
                self.Cx = self.optC()
                self.e3 = self.dist(tmp3, self.Cx)
            else:
                print("C OK and e3 is ", self.e3)


            d =  max((self.e1,self.e2,self.e3))
            print('fraction rate = ',d)
            if d <= self.paramkey['ITER_TOL'] :
                break

        tmp = self.Bx.dot(np.transpose(self.Ax)) + self.Ax.dot(np.transpose(self.Bx))
        tmp = tmp /2

        val,vec = np.linalg.eig(tmp)
        ind = val.argsort()
        ind = ind [::-1]
        vec = vec[:,ind]

        Q = vec[:,0:self.m_bar]

        self.sol = np.transpose(Q)

        Q = np.transpose(Q)
        np.save( os.path.join(self.paramkey['SAVE_PATH'],'Q.npy'),Q)

        return Q

    def multi_sol(self):

        Q1 = self.solution(5)
        m_bar_advice = np.int( np.ceil( np.mean(self.mList) ) )
        m_bar_advice = max((self.m_bar)+1,m_bar_advice)
        Q2 = self.solution(m_bar_advice = m_bar_advice)
        return Q1,Q2

    def get_init(self,m_bar_advice = 0):

        self.P = (self.P+np.transpose(self.P))/2
        eigvalue,eigvector = np.linalg.eig(self.P)
        idx = eigvalue.argsort()
        idx = idx[::-1] # 数组调过来改为降序

        # 获得排序后的特征值和特征向量
        eigvalue = eigvalue[idx]
        eigvector = eigvector[:,idx]

        # 求最大的gap来确定m_bar
        l = len(eigvalue)
        dv = eigvalue[0:l-1] - eigvalue[1:l]
        print("eigvalue",eigvalue)
        m_bar = dv.argmax()+1 # 因为python是从零开始标的
        m_bar = max(m_bar,m_bar_advice)

        # 获得AB初值
        tmp  = np.dot( eigvector[:,0:m_bar] , np.diag(eigvalue[0:m_bar])**0.5 )
        #tmp = np.random.random(np.shape(tmp))
        Ctmp = np.ones([self.n,m_bar])

        return m_bar, tmp, tmp, Ctmp

    ######### Optimize ###########
    def optA(self):

        # 公式(17)
        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)
        for r in range(self.N):
            for t in range(self.n):
                B_tmp = self.Wrst[r,:,t]
                B_tmp = self.Bx.transpose() * B_tmp
                B_tmp = B_tmp.dot(self.Bx)
                C = self.Cx[t]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[r][i][j] += B_tmp[i][j]*C[i]*C[j]

        g = np.zeros([self.N,self.m_bar],np.double)

        for r in range(self.N):
            for s in range(self.N):
                C_tmp = np.multiply( self.Wrst[r][s], self.R[r][s] )
                C_tmp = np.dot(C_tmp,self.Cx)
                g[r] += np.multiply( self.Bx[s],C_tmp )

        ans = np.zeros([self.N,self.m_bar])


        for r in range(self.N):

            if(np.linalg.cond(H[r]))  > self.paramkey['REG_TOL']:
                H[r] += self.paramkey['REG_ADD'] * np.diag( np.ones(np.shape(H[r])[0]) )
            #print(H[r])
            ans[r] = np.linalg.pinv(H[r],self.paramkey['PINV_TOL']).dot(g[r])
        #print(ans,self.Ax)
        return ans

    def optB(self):

        # 公式(19) ctrl c ctrl v 不规范，亲人两行泪

        H = np.zeros([self.N,self.m_bar,self.m_bar],np.double)

        for s in range(self.N):
            for t in range(self.n):
                A_tmp = self.Wrst[:,s,t]
                A_tmp = self.Ax.transpose() * A_tmp
                A_tmp = A_tmp.dot(self.Ax)
                C = self.Cx[t]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[s][i][j] += A_tmp[i][j]*C[i]*C[j]

        g = np.zeros([self.N,self.m_bar],np.double)


        for s in range(self.N):
            for r in range(self.N):
                C_tmp = np.multiply(self.Wrst[r][s], self.R[r][s])
                C_tmp = np.dot(C_tmp, self.Cx)
                g[s] += np.multiply(self.Ax[r], C_tmp)

        ans = np.zeros([self.N,self.m_bar])

        for s in range(self.N):

            if (np.linalg.cond(H[s])) > self.paramkey['REG_TOL']:
                H[s] += self.paramkey['REG_ADD']* np.diag(np.ones(np.shape(H[s])[0]))

            ans[s] = np.linalg.pinv(H[s],self.paramkey['PINV_TOL']).dot(g[s])

        return ans

    def optC(self):

        # 公式(20) 这个公式大小又双叒叕写错了
        H = np.zeros([self.n,self.m_bar,self.m_bar],np.double)

        for t in range(self.n):
            for r in range(self.N):
                B_tmp = self.Wrst[r, :, t]
                B_tmp = self.Bx.transpose() * B_tmp
                B_tmp = B_tmp.dot(self.Bx)
                A = self.Ax[r]
                for i in range(self.m_bar):
                    for j in range(self.m_bar):
                        H[t][i][j] += B_tmp[i][j] * A[i] * A[j]


        g = np.zeros([self.n,self.m_bar],np.double)


        for t in range(self.n):
            for s in range(self.N):
                A_tmp = np.multiply( self.Wrst[:,s,t], self.R[:,s,t] )
                A_tmp = np.dot(A_tmp,self.Ax)
                g[t] += np.multiply( self.Bx[s],A_tmp )
        ans = np.zeros([self.n,self.m_bar])

        for t in range(self.n):
            if (np.linalg.cond(H[t])) > self.paramkey['REG_TOL']:
                H[t] += self.paramkey['REG_ADD'] * np.diag(np.ones(np.shape(H[t])[0]))

            ans[t] = np.linalg.pinv(H[t],self.paramkey['PINV_TOL']).dot(g[t])

        return ans

    ######### Round ##########
    def rounded_solution(self, th=0.5, k = 1 ,sol=None):

        '''
        :param th: 两个向量大于th算是一类
        :param k: 允许一个universal part 对应到某个物体的k个part
        :param t: 允许一个part对应到t个universal part
        :param sol: degbug用的，输入一个现成的小数解他就不用自己再求一个
        :return:
        '''

        if sol is None:
            sol = self.solution()

        sol = np.transpose(sol)
        n, m = np.shape(sol)

        for r in range(n):
            if(sum(sol[r]**2)**0.5 ==0):
                    print('zero',r)
                    continue
            sol[r] = sol[r] / sum(sol[r]**2)**0.5

        N = np.cumsum(self.mList)
        flag = np.zeros([n], np.int)
        ans = np.zeros([n, 0], np.int)
        for r in range(n):
            if (flag[r] != 0):
                continue

            cur_ans = np.zeros([n, 1], np.int)
            cur_ans[r] = 1
            flag[r] = 1

            ob = np.where(N > r)
            ob = np.min(ob)

            cur_center = sol[r, :]
            if ob < self.n:
                for ob1 in range(ob + 1, self.n):

                    cc = np.dot(cur_center, np.transpose(sol[N[ob1 - 1] : N[ob1], :]))
                    q = flag[ N[ob1 - 1]:N[ob1] ]
                    l = len(cc)

                    mind = []

                    for ind in range(l):
                        if  q[ind] == 0:
                            mind.append( ind )
                    mind.sort(key= lambda i: cc[i],reverse=True)
                    if(len(mind) > k):
                        mind = mind[0:k]


                    for idx in range( len(mind) ):
                        i = mind[idx]
                        if cc[i] < th:
                            break
                        f = 1

                        # compare with others
                        for jdx in range(idx):
                            j = mind[jdx]
                            if np.dot( sol[j + N[ob1-1] ,: ] , sol[i +N[ob1-1] ,: ]) < th:
                                f = 0
                                break
                        if(f ==0 ):
                            continue

                        cur_ans[N[ob1 - 1] + i] = 1
                        flag[N[ob1 - 1] + i] = 1

            ans = np.hstack([ans, cur_ans])
        self.rounded_sol = ans.copy()
        return ans

    def multi_round(self, th = 0.5, k = 2, multi_sol = None):

        if multi_sol is None:
            multi_sol = self.multi_sol()
        #print('shape:' ,np.shape(multi_sol[0]),np.shape(multi_sol[1]) )
        sol,sol_advice = np.transpose(multi_sol[0] ) , np.transpose(multi_sol[1])

        n, m = np.shape(sol)

        for r in range(n):
            if(sum(sol[r]**2)**0.5 ==0):
                    continue
            sol[r] = sol[r] / sum(sol[r]**2)**0.5
            if (sum(sol_advice[r] ** 2) ** 0.5 == 0):
                continue
            sol_advice[r] = sol_advice[r] / sum(sol_advice[r] ** 2) ** 0.5

        N = np.cumsum(self.mList)
        flag = np.zeros([n], np.int)
        ans = np.zeros([n, 0], np.int)
        for r in range(n):
            if (flag[r] != 0):
                continue

            cur_ans = np.zeros([n, 1], np.int)
            cur_ans[r] = 1
            flag[r] = 1

            ob = np.where(N > r)
            ob = np.min(ob)

            cur_center = sol[r, :]
            cur_center_advice = sol_advice[r, :]
            if ob < self.n:
                for ob1 in range(ob + 1, self.n):

                    cc = np.dot(cur_center, np.transpose(sol[N[ob1 - 1] : N[ob1], :]))
                    q = flag[ N[ob1 - 1]:N[ob1] ]
                    l = len(cc)

                    mind = []

                    for ind in range(l):
                        if  q[ind] == 0:
                            mind.append( ind )
                    mind.sort(key= lambda i: cc[i],reverse=True)
                    if(len(mind) > k):
                        mind = mind[0:k]

                    if(len(mind) > 1):
                        mind_tmp = [mind[0]]
                        for ii in range(1,len(mind)):
                            if(np.dot( cur_center_advice, sol_advice[ N[ob1-1] + mind[ii] ,: ] ) > th ):
                                mind_tmp.append(mind[ii])
                        mind = mind_tmp

                    for idx in range( len(mind) ):
                        i = mind[idx]
                        if cc[i] < th:
                            break
                        f = 1

                        # compare with others
                        for jdx in range(idx):
                            j = mind[jdx]
                            if np.dot( sol[j + N[ob1-1] ,: ] , sol[i +N[ob1-1] ,: ]) < th:
                                f = 0
                                break
                        if(f ==0 ):
                            continue

                        cur_ans[N[ob1 - 1] + i] = 1
                        flag[N[ob1 - 1] + i] = 1

            ans = np.hstack([ans, cur_ans])
        self.rounded_sol = ans.copy()
        return ans
    ######### Tools ##########

    def set_param(self, param):
        '''
        :param param: [key, value, key ,value ,......]
        :return:
        '''
        n = len(param)
        rear = 0

        self.paramkey = {'ITER_TOL':1e-12,
                         'ITER_DISFUNC':self.dist_max,
                         'ITER_MAXNUM':100,
                         'REG_TOL':100,
                         'REG_ADD':6,
                         'PINV_TOL':1e-10,
                         'SAVE_TMP':True,
                         'SAVE_PATH': 'TmpData/'
                         }

        while rear < n :
            str = param[rear]
            if str=='ITER_DISFUNC':
                self.paramkey['ITER_DISFUNC'] = self.dist_parse(param[rear+1])
            else :
                self.paramkey[str] = param[rear+1]
            rear+=2

        return

    def dist_parse(self,dist_name):
        dist_key = { 'DIST_MAX': self.dist_max,
                     0: self.dist_max,
        }
        return dist_key[dist_name]

    def dist_max(self, A, B):
        return np.max(np.abs(A-B) )


class TensorMapVis:
    def __init__(self, image_list, mList, rouned_sol):

        w,h = 256,256

        data_num = np.size( mList)

        color_num = np.size(rouned_sol[0])
        cmap = plt.get_cmap('gist_ncar')
        colors_set = cmap(np.linspace(0, 1, color_num + 1))
        image_label_index = np.zeros([data_num, color_num], np.int)

        image_list_old = image_list

        new_image_label_list = []
        now_n = 0

        self.draw_image = []

        for image_index in range(data_num):
            this_image_old_label = image_list_old[image_index]
            this_image_part_nums = int(this_image_old_label.max())
            # print(this_image_part_nums)
            temp_part_index = []
            new_this_image = np.zeros([w, h], np.ndarray)
            for part_index in range(1, this_image_part_nums + 1):
                this_part_map = rouned_sol[now_n]
                now_n += 1
                this_part_new_label = np.where(this_part_map == 1)
                if (np.size(this_part_new_label) > 1):
                    print('>1:', this_part_map)
                else:
                    print('<=1:', this_part_map)

                this_part_new_label = np.array([cc for cc in this_part_new_label])
                # print('????',  np.ndarray( [cc[0] for cc in this_part_new_label] ) )
                temp_part_index.append(this_part_new_label)
                # print('part',this_part_new_label)

                tmp_ind = np.where(this_image_old_label == part_index)
                l = np.shape(tmp_ind)[1]
                for ii in range(l):
                    new_this_image[tmp_ind[0][ii]][tmp_ind[1][ii]] = this_part_new_label + 1
                # print('this_part',this_part_new_label+1)
            new_image_label_list.append(new_this_image)
        # print('$',type(new_image_label_list[0][0][0]) )

        for image_index in range(data_num):
            final_im = new_image_label_list[image_index]
            draw_image = np.zeros([w, h, 3])
            # max_color = int(final_im.max())
            myrand = [np.random.randint(0, i + 1) for i in range(10)]
            for i in range(w):
                if i % 10 == 0:
                    myrand = [np.random.randint(0, i + 1) for i in range(10)]
                for j in range(h):
                    c = final_im[i][j]
                    if type(c) != np.ndarray:
                        continue
                    # print('c',c)
                    # if(np.size(c) >1):
                    # print('c>1:',c)
                    t = myrand[np.size(c[0]) - 1]
                    t = c[0][t]
                    # print(t)
                    draw_image[i][j] = colors_set[t][0:3] * 255
            draw_image = draw_image.astype(np.uint8)
            self.draw_image.append(draw_image)

