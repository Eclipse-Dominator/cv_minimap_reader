import numpy as np
import cv2
import time
class MapReader:
    def __init__(self,map):
        '''
        index_params= dict(algorithm = 6,table_number = 6, key_size = 12, multi_probe_level = 1) 
        self.descriptor = cv2.ORB_create(nfeatures=100000,scoreType=cv2.ORB_HARRIS_SCORE)
        '''
        
        self.descriptor = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=10)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        
        
        #self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        self.getkpdes = lambda x: self.descriptor.detectAndCompute(x,None)
        self.kp,self.des = self.getkpdes(map)


    def getPTFromImageToMap(self,minimap,PT):
        kp,des = self.getkpdes(minimap)
        
        matches = self.matcher.knnMatch(des,self.des,k=2)
        oc_cumulative =np.zeros((2,))
        oc_confidence = np.zeros((2,))
        oc_counter = 0
        oa = OA = AC = None
        OC = np.array(PT)
        angle_cumulative = np.array([0.0,1.0])
        for m,n in matches:
            if m.distance > 0.5*n.distance:
                continue

            ob = np.array(self.kp[m.trainIdx].pt)
            OB = np.array(kp[m.queryIdx].pt)
            if oa is None:
                oa = ob
                OA = OB
                AC = OC - OA
                continue
            oc,angle = self.transformationPT(OA,OB,oa,ob,AC)
            angle_cumulative += angle
            #oc = self.transformPTinMap(OA,OB,AC,oa,ob)
            
            if oc_counter>10:
                oc_confidence += oc - oc_cumulative/oc_counter
            if oc_counter>30:
                average_confidence = oc_confidence/(oc_counter-10)
                
                if (np.abs(average_confidence)<3).all():
                    break
            
            oc_cumulative += oc
            oc_counter+=1
        if oc_counter:
            return oc_cumulative/oc_counter,angle_cumulative/oc_counter
        return None,None
    
    
    def transformationPT(self,OA,OB,oa,ob,AC): # returns coordinate and angle difference from map
        ab_d = self.distance(ob,oa)
        AB_d = self.distance(OB,OA)
        scale_factor = ab_d/AB_d
        AB_u = (OB-OA) / AB_d
        ab_u = (ob-oa) / ab_d
        cosx = AB_u[0]*ab_u[0] + AB_u[1]*ab_u[1]
        sinx = AB_u[0]*ab_u[1] - AB_u[1] * ab_u[0]
        return oa + np.matmul([[cosx,-sinx],[sinx,cosx]],AC)*scale_factor,[-sinx,cosx]

    def transformPTinMap(self,OA,OB,AC,oa,ob): # no map rotation
        # suppose coordinate OA,OB,OC in new scaled map oa,ob,oc
        # input are numpy (2,) representing column vectors
        scale_factor = self.distance(ob,oa)/self.distance(OB,OA)
        return oa+ AC*scale_factor

    @staticmethod
    def distance(pt1,pt2):
        return np.sqrt(np.sum(np.square(pt1-pt2)))+0.0000000001 # ensure no zero division