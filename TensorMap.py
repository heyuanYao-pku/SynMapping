import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import os


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



        if Not_Build==False: # Check Inputs

            assert np.size(part_num_list) == shape_num, 'Length of part num list is not equal to shape num\n'
            assert np.shape(Map_list) == (shape_num,shape_num), 'Shape of Map_list is not equal to (shape num, shape num)\n'
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

    def buildC(self):
        '''
        计算 Cijk =  Pki * Pjk * Pij 并舍弃对角线以外的值
        '''
        n = self.n
        self.Clist = np.zeros([n,n,n],np.ndarray)
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


    def solution(self):

        self.build_Wrst()

        self.m_bar,self.Ax,self.Bx,self.Cx = self.get_init()
        print('m_bar = ',self.m_bar)
        self.Ax = np.array(self.Ax,np.double)
        self.Bx = np.array(self.Bx, np.double)
        self.Cx = np.array(self.Cx, np.double)


        cont  = 0

        while(True):

            tmp1, tmp2,tmp3 = self.Ax.copy(),self.Bx.copy(),self.Cx.copy()

            cont +=1
            print("iter%d"%cont)
            if cont == self.paramkey['ITER_MAXNUM']:
                break
            self.Ax = self.optA()
            self.Bx = self.optB()
            self.Cx = self.optC()

            self.dist = self.paramkey['ITER_DISFUNC']

            d = self.dist(tmp1,self.Ax) + self.dist(tmp2,self.Bx) + self.dist(tmp3,self.Cx)
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

        return np.transpose(Q)

    def dist_max(self, A, B):
        return np.max(np.abs(A-B) )

    def get_init(self):

        eigvalue,eigvector = np.linalg.eig(self.P)
        idx = eigvalue.argsort()
        idx = idx[::-1] # 数组调过来改为降序

        # 获得排序后的特征值和特征向量
        eigvalue = eigvalue[idx]
        eigvector = eigvector[:,idx]

        # 求最大的gap来确定m_bar
        l = len(eigvalue)
        dv = eigvalue[0:l-1] - eigvalue[1:l]
        m_bar = dv.argmax()+1 # 因为python是从零开始标的

        # 获得AB初值
        tmp  = np.dot( eigvector[:,0:m_bar] , np.diag(eigvalue[0:m_bar])**0.5 )

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
            ans[r] = np.linalg.pinv(H[r],self.paramkey['PINV_TOL']).dot(g[r])

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

            ans[s] = np.linalg.pinv(H[s],'PINV_TOL').dot(g[s])

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

    ######### Tools ##########

    def set_param(self, param):
        n = len(param)
        rear = 0
        self.paramkey = {'ITER_TOL':1e-3,
                         'ITER_DISFUNC':self.dist_max,
                         'ITER_MAXNUM':100,
                         'REG_TOL':100,
                         'REG_ADD':3,
                         'PINV_TOL':10
                         }
        while rear < n :
            str = param[rear]
            if str=='ITER_DISFUNC':
                self.paramkey['ITER_DISFUNC'] = self.dist_parse(param[rear+1])
            else :
                self.paramkey[str] = param[rear+1]
            rear+=2

    def dist_parse(self,dist_name):
        dist_key = { 'DIST_MAX': self.dist_max,
                     0: self.dist_max,
        }
        return dist_key[dist_name]
