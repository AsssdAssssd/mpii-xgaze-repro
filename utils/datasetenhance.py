import cv2
import numpy as np
import random
LUMA_W = np.array([0.0722,0.7152,0.2126],dtype=np.float32) #bgr
class DatasetEnhancer:

    def __init__(self,rate):
        self.rate = rate #有多少概率增强，相当于实际的样本数变成了（1+rate）*n 
        self.gamma_range=0.2
        self.wb_range=0.3
        self.exposure_ev_range=[-2.0,0.5] #EV
        self.p_exposure=0.8
        self.p_gamma=0.5
        self.p_directional_gradient = 0.5
        self.p_wb = 0.5
        self.grad_map = 0.8 #斜着光线最大增益


    def whether_enhance(self):
        if np.random.rand() < self.rate:
            return True
        return False
    
    def image_light_enhance(self,img):
        img = img.astype(np.float32)/255.0 #0-1
        img=self._exposure(img)
        img=self._gamma(img)
        img=self._directional_gradient(img)
        img=self._white_balance(img)
        return np.clip(img*255.0,0.0,255.0).astype(np.uint8)


    def _exposure(self,img):
        if np.random.rand() > self.p_exposure:
            return img
        ev = random.uniform(self.exposure_ev_range[0], self.exposure_ev_range[1])
        img = img * (2**ev)
        return img
    
    def _gamma(self,img):
        if np.random.rand() > self.p_gamma:
            return img
        g = 1.0 + random.uniform(-self.gamma_range,self.gamma_range)
        return  np.power(np.clip(img,0.0,1.0),1.0/g)
    
    def _directional_gradient(self,img):
        if np.random.rand() > self.p_directional_gradient:
            return img
        h,w=img.shape[:2]
        ang = random.uniform(0.0,2.0*np.pi)
        ys,xs = np.mgrid[0:h,0:w].astype(np.float32) #得到坐标矩阵
        proj = (xs/w - 0.5)*np.cos(ang) + (ys/h -0.5)*np.sin(ang) #距离中心*光线方向
        proj /=(np.abs(proj).max()+1e-6) #-1,1
        amp = random.uniform(0.4,self.grad_map)
        return img*(1.0+amp*proj)[...,None]

    def _white_balance(self,img): #默认读取进来是bgr
        if np.random.rand() > self.p_wb:
            return img
        gains=np.array([1.0 + random.uniform(-self.wb_range,self.wb_range)for _ in range(3)],dtype=np.float32)
        gains /=(gains@LUMA_W)
        return img*gains

if __name__  =="__main__":
    img =cv2.imread('/home/zhongrui/datasets/GazeCapture/00003/frames/00031.jpg')
    dataenhance=DatasetEnhancer(1.0)
    wind= cv2.namedWindow("tmp")
    cv2.imshow("tmp",dataenhance.image_light_enhance(img))
    cv2.waitKey()

