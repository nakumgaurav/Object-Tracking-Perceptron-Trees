#!/usr/bin/python

# import vot
import os
import sys
import time
import cv2
import numpy as np
import collections
import math
import random
import pickle
from tqdm import tqdm
import object_track2 #import compute_fitness_score

Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

class PKTracker(object):

    def __init__(self, image, region):
       
        cen_x=0
        cen_y=0
        min_x=1000
        min_y=1000
        max_x=-1
        max_y=-1
        for p in region.points:
            #print p.x
            #print p.y
            cen_x+=p.x
            cen_y+=p.y
            min_x=min(min_x,p.x)
            max_x=max(max_x,p.x)
            min_y=min(min_y,p.y)
            max_y=max(max_y,p.y)            
        cen_x=int(cen_x*0.25)
        cen_y=int(cen_y*0.25)
        
        self.trans_x=0
        self.trans_y=0        
        
        obj_width=max_x-min_x
        obj_height=max_y-min_y
        #print "other"
        #print cen_x
        #print cen_y
        #print obj_width
        #print obj_height
        #print max_x
        #print min_x
        #print max_y
        #print min_y
        self._set=0
        rr,cc=image.shape[:2]

       
        
        # self.trans_x=0
        # self.trans_y=0

       
        
        
        if( obj_width%2 == 0 ):
            obj_width+= 1
            
        if( obj_height %2 ==0 ):
            obj_height +=1    
            
            
        self._curr_width=int(obj_width)
        self._curr_height=int(obj_height)
    
        self._tar_width=int(obj_width)
        self._tar_height=int(obj_height)        
    
        self._curr_half_width = int( ( self._curr_width  -1 ) *0.5 )
        self._curr_half_height = int( ( self._curr_height  -1 ) *0.5 )        
    
        # self._similarity_BC=0.0
        self._score_ = 0.0
    
        # specification for the features
        self._bins_per_channel = 16
        self._bin_size = int( np.floor( 256 /self._bins_per_channel) )
        self._model_dim = np.power(self._bins_per_channel , 3 )        
    
        self._curr_angle=0.0
    
        self._target_model = np.zeros( self._model_dim )
        self._prev_model = np.zeros( self._model_dim )
        self._curr_model = np.zeros( self._model_dim )
        self._out_model = np.zeros( self._model_dim )
    
        self._curr_centrex=0
        self._curr_centrey=0
    
        self._prev_centrex=cen_x
        self._prev_centrey=cen_y       
    
        self._centre_vel_x=0
        self._centre_vel_y=0
    
    
        self._curr_centroid_x=cen_x
        self._curr_centroid_y=cen_y            
    
        self._maxItr=10
        self._popSize=25
        #print self._popSize
        self._Sx=np.array([[0.7,1.3]])
        self._Sy=np.array([[0.7,1.3]])
        # self._breedRatio=0.3
        self._angrange=np.array([[0,180]])

        self._similarity_BC  =0.0
        
        self._X=np.zeros((5,self._popSize))
        self._Z=np.zeros((1,self._popSize))
        # self._discard=int(self._popSize*self._breedRatio)
        self._prev_P=np.zeros((5,self._popSize))
        self.combined_index=np.zeros([int(self._curr_height),int(self._curr_width)])   
        self.compute_target_model(image)
    
    def check1(self,X):
        X[0]=max(X[0],self.trans_x[0,0])
        X[0]=min(X[0],self.trans_x[0,1])
        X[1]=max(X[1],self.trans_y[0,0])
        X[1]=min(X[1],self.trans_y[0,1])
    
        X[2]=max(X[2],self._Sx[0,0])
        X[2]=min(X[2],self._Sx[0,1])
    
        X[3]=max(X[3],self._Sy[0,0])
        X[3]=min(X[3],self._Sy[0,1])   
    
        X[4]=max(X[4],self._angrange[0,0])
        X[4]=min(X[4],self._angrange[0,1])    
        return X;        
        
    def check(self,X):
        a,b=X.shape
        for i in range(b):
            X[0,i]=max(X[0,i],self.trans_x[0,0])
            X[0,i]=min(X[0,i],self.trans_x[0,1])
            X[1,i]=max(X[1,i],self.trans_y[0,0])
            X[1,i]=min(X[1,i],self.trans_y[0,1])
        
            X[2,i]=max(X[2,i],self._Sx[0,0])
            X[2,i]=min(X[2,i],self._Sx[0,1])
        
            X[3,i]=max(X[3,i],self._Sy[0,0])
            X[3,i]=min(X[3,i],self._Sy[0,1])   
        
            X[4,i]=max(X[4,i],self._angrange[0,0])
            X[4,i]=min(X[4,i],self._angrange[0,1])    
        return X;    
            
    def compute_ellipse_kernel( self ):
        """ compute the ellipse kernel weights 
        """
        
        error_code = 0 
        
        if( int(self._curr_width) %2 == 0 ):
            f_width=int(self._curr_width) + 1  
            keyx=1
        else:
            f_width=int(self._curr_width)  
            keyx=0
            
        if( int(self._curr_height) %2 == 0 ):
            f_height=int(self._curr_height) + 1  
            keyy=1
        else:
            f_height=int(self._curr_height)  
            keyy=0        
        
        half_width = int(( f_width -1 ) * 0.5)
        half_height =int( ( f_height -1 ) * 0.5)        
        
        
        x_limit = int( np.floor( ( f_width - 1)  * 0.5 ) )
        
        y_limit = int( np.floor( ( f_height -1 ) * 0.5 ) )
        
        x_range = np.array( [ range( -x_limit , x_limit -keyx+ 1 )])
        y_range = np.array( [ range( -y_limit , y_limit -keyy+ 1 ) ])
        y_range = np.transpose( y_range)
        x_matrix = np.repeat( x_range , y_limit * 2 -keyy+ 1 , axis = 0 )
        y_matrix = np.repeat( y_range , x_limit*2 -keyx + 1 , axis = 1 )
        
        x_square = np.multiply( x_matrix , x_matrix )
        y_square = np.multiply( y_matrix ,y_matrix ) 
        #print "xlimit" ,x_limit
        #print "ylimit", y_limit
        #print "current width",self._curr_width
        #print "current height",self._curr_height
        
        #print "xmatrix shape",x_matrix.shape
        #print "ymatrix shape",y_matrix.shape
        
        x_square  = np.divide( x_square , float( half_width * half_width ) )
        y_square  = np.divide( y_square , float( half_height * half_height ) )
        
        self._kernel_mask  = np.ones( [ int(self._curr_height) , int(self._curr_width) ] ) -  ( y_square + x_square )
        
        self._kernel_mask [ self._kernel_mask < 0 ] = 0
        
        # print( 'kernel computation complete ')
        
        return error_code  
    def compute_target_model( self, image  ):
        
        error_code = 0
        self._set=1
        self.compute_object_model(image )
        
        self._target_model = np.copy( self._curr_model )
       
        # print( 'Target model computation complete')
        return error_code
    
    
    
    def takeout(self,image,width,height):
        v_x = (math.cos(self._curr_angle), math.sin(self._curr_angle))
        v_y = (-math.sin(self._curr_angle), math.cos(self._curr_angle))
        s_x = int(self._curr_centroid_x - v_x[0] * (width / 2) - v_y[0] * (height / 2))
        s_y = int(self._curr_centroid_y - v_x[1] * (width / 2) - v_y[1] * (height / 2))

        mapping = np.array([[v_x[0],v_y[0], s_x],
                            [v_x[1],v_y[1], s_y]])
        #print mapping
        #print self._curr_width
        #print self._curr_height
        #print self._X
        #self._curr_width=abs(self._curr_width)
        #self._curr_height=abs(self._curr_height)
        return cv2.warpAffine(image,mapping,(int(width), int(height)),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)         
    
    def compute_object_model( self , image ):
        self._curr_model = self._curr_model * 0.0
        
        self.combined_index = self.combined_index * 0
        
        output_image_in=self.takeout(image,self._curr_width,self._curr_height)
        output_image_out=self.takeout(image,int(math.sqrt(2)*self._curr_width),int(math.sqrt(2)*self._curr_height))
        output_image_in=np.asarray(output_image_in[:,:])
        output_image_out=np.asarray(output_image_out[:,:])

        index_matrix =  np.divide( output_image_in , self._bin_size )
        index_matrix = np.floor( index_matrix )
        index_matrix  = index_matrix.astype( int )

        b_index, g_index, r_index  = cv2.split( index_matrix )

        combined_index = b_index * np.power( self._bins_per_channel ,2 )  +  self._bins_per_channel  * g_index + r_index 
        combined_index = combined_index.astype( int )


        index_matrix_out =  np.divide( output_image_out , self._bin_size )
        index_matrix_out = np.floor( index_matrix_out )
        index_matrix_out  = index_matrix_out.astype( int )

        bb_index, gg_index, rr_index  = cv2.split( index_matrix_out )

        combined_index_out = bb_index * np.power( self._bins_per_channel ,2 )  +  self._bins_per_channel  * gg_index + rr_index 
        combined_index_out = combined_index_out.astype( int )        

        z1=int((math.sqrt(2)-1)*0.5*len(output_image_in))
        z2=int((math.sqrt(2)-1)*0.5*len(output_image_in[0]))
        s=[]
        for i in range(len(output_image_out)):
            for j in range(len(output_image_out[0])):
                if (i > z1 and i < len(output_image_out)-z1 and j< z2) or (i > z1 and i < len(output_image_out)-z1 and j > len(output_image_out[0])-z2) or (i<z1) or (i> len(output_image_out)-z1):
                    s.append(combined_index_out[i,j])

        self._out_model=np.bincount(s,minlength=16**3)


        self.combined_index=np.zeros([self._curr_height,self._curr_width])
        self.combined_index = combined_index.astype( int )
        self.compute_ellipse_kernel()
        #print( self._curr_model.shape )
        for i in range ( 0 , int(self._curr_height) ):
            for j in range( 0, int(self._curr_width) ):
                self._curr_model[ combined_index[ i , j ] ] +=  self._kernel_mask[ i,j] 


        #l1 normalize the feature( histogram )
        sum_out=np.sum(self._out_model)
        sum_val = np.sum( self._curr_model)
        self._curr_model = self._curr_model/float( sum_val)
        self._out_model=self._out_model/float(sum_out)
        ans1=self._curr_model
        ans2=self._out_model
        if self._set==1:
            for i in range(len(self._curr_model)):
                if ans1[i]!=0 and ans2[i]!=0 and ans1[i]<ans2[i]:
                    self._curr_model[i]=0

        # print('Object model computed ')      


    def compute_fitness_value( self, image, forests, classifier, fixed_shape ):
        """ 
            compute the fitness function
        """

        patch_data = (self._curr_centroid_x, self._curr_centroid_y, self._curr_width, self._curr_height, self._curr_angle)
        # self._score_ = object_track2.compute_fitness_score(classifier, forests, image, patch_data, fixed_shape)


        self._similarity_BC  =0.0
        BC1=0.0
        BC2=0.0
        # Bhattacharya similariy between two distributions
        for i in range( self._model_dim ):
            if( self._target_model[i] !=0 and self._curr_model[i] != 0 ):
                BC1 +=  (np.sqrt( self._target_model[i] *  self._curr_model[i] ))
                BC2+= (np.sqrt(self._target_model[i] *  self._out_model[i]))
        # print(BC1)
        # print(BC2)
        self._similarity_BC = BC1*(1-BC2)
        # print(self._score_)

    def pksort(self,T):
        a,n=T.shape
        for i in range(0,n-1):
            for j in range(0,n-i-1):
         
                if T[0,j]<T[0,j+1]:
                    c0=T[0,j+1]
                    T[0,j+1]=T[0,j]
                    T[0,j]=c0

                    c1=T[1,j+1]
                    T[1,j+1]=T[1,j]
                    T[1,j]=c1

                    c2=T[2,j+1]
                    T[2,j+1]=T[2,j]
                    T[2,j]=c2

                    c3=T[3,j+1]
                    T[3,j+1]=T[3,j]
                    T[3,j]=c3

                    c4=T[4,j+1]
                    T[4,j+1]=T[4,j]
                    T[4,j]=c4
                    
                    c5=T[5,j+1]
                    T[5,j+1]=T[5,j]
                    T[5,j]=c5            
                    
        return T; 
    
    
    # def DEcomp(self,image,X):
    #     a,b=X.shape
    #     Y=np.zeros((a,b))
    #     xx=np.arange(b)
    #     for i in range(b):
    #         while(1):
    #             j1=random.choice(xx)
    #             j2=random.choice(xx)
    #             j3=random.choice(xx)
    #             if j1!=j2!=j3!=i:
    #                 break
    #         Y[:,i]=X[:,j1]+0.9*(X[:,j2]-X[:,j3])
            
    #         Y[:,i]=self.check1(Y[:,i])
    #         if np.random.rand()>0.9:
    #             Y[:,i]=X[:,i]
            
    #         dd1=self.calcSMobo(image,Y[:,i])
    #         dd2=self.calcSMobo(image,X[:,i])
    #         if dd1 < dd2:
    #             Y[:,i]=X[:,i]
        
    #     return Y 
    
    
    def popx( self,n):
        x=np.arange(self.trans_x[0,0],self.trans_x[0,1]+0.5,1)
        y=np.arange(self.trans_y[0,0],self.trans_y[0,1]+0.5,1)
        sx=np.arange(self._Sx[0,0],self._Sx[0,1]+0.05,0.1)
        sy=np.arange(self._Sy[0,0],self._Sy[0,1]+0.05,0.1)
        ang=np.arange(self._angrange[0,0],self._angrange[0,1]+0.5,1)
        
        for i in range(0,n):
            self._X[0,i]=random.choice(x)
            self._X[1,i]=random.choice(y)
            self._X[2,i]=random.choice(sx)
            self._X[3,i]=random.choice(sy)
            self._X[4,i]=random.choice(ang)
    
    def calcSM(self,image, forests, classifier, fixed_shape):
        a,n=self._X.shape
        self._Z=np.zeros((1,n))
        for i in range(0,n):
            self._curr_centroid_x=self._X[0,i]
            self._curr_centroid_y=self._X[1,i]
            self._curr_width=int(self._X[2,i]*self._tar_width)
            self._curr_height=int(self._X[3,i]*self._tar_height)
            self._curr_angle=self._X[4,i]
            self._curr_angle=self._curr_angle*(np.pi/180)
            self.compute_object_model( image )
            self.compute_fitness_value(image, forests, classifier, fixed_shape)
            self._Z[0,i]=self._score_
            self._Z[0,i]=self._similarity_BC 

        def normalize(vec):
            vec += np.abs(np.min(vec))
            # print("vec=",vec)
            assert np.max(vec) > 0
            vec /= np.max(vec)

        normalize(self._Z[0])
    # def calcSMobo(self,image,X):
        
        
    #     self._curr_centroid_x=X[0]
    #     self._curr_centroid_y=X[1]
    #     self._curr_width=int(X[2]*self._tar_width)
    #     self._curr_height=int(X[3]*self._tar_height)
    #     self._curr_angle=X[4]
    #     self._curr_angle=self._curr_angle*(np.pi/180)
    #     self.compute_object_model( image )
    #     self.compute_similarity_value()
    #     return self._similarity_BC;       
            
    def updatePosition(self,Xin,Vin):
        a,n=Xin.shape
        P=np.zeros((a,n))
        for i in range(0,n):
            P[0,i]=int(Xin[0,i]+Vin[0,i])
            P[1,i]=int(Xin[1,i]+Vin[1,i])
            P[2,i]=np.round((Xin[2,i]+Vin[2,i]),1)
            P[3,i]=np.round((Xin[3,i]+Vin[3,i]),1)
            P[4,i]=int((Xin[4,i]+Vin[4,i]))
        return P;    
    
    
    def updateVelocity(self,Xout,Vold,pbest,gbest):
        a,b=Xout.shape
        Vnew=np.zeros((a,b))

        phi1=np.random.rand()
        phi2=np.random.rand()
        c1=c2=2.3
        p=c1+c2
        K=(2/abs(2-p-np.sqrt(p**2-4*p)))

        gg=np.array([gbest,]*b).transpose()
        Vnew=K*(Vold+c1*phi1*(pbest-Xout)+ c2*phi2*(gg-Xout))

        return Vnew;
    
    
    
    
    def rotate(self,origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
    
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point
    
        qx = int(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (oy - py))
        qy = int(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (oy - py))
        return qx, qy 
    
    def generateOutput(self):
        x=int(self._curr_centroid_x)
        y=int(self._curr_centroid_y)
        width=int(self._curr_width*self._tar_width)
        height=int(self._curr_height*self._tar_height)
        angle=int(self._curr_angle)
        self.trans_x=np.array([[x-30,x+30]])
        self.trans_y=np.array([[y-30,y+30]])
        #print self._curr_width
        #print self._curr_height
        #print self._tar_width
        #print self._tar_height
        # print(x)
        # print(y)
        # print(width)
        # print(height)
        # print(angle)
        xmax=abs(x+round(width/2))
        xmin=abs(x-round(width/2))
    
        ymax=abs(y+round(height/2))
        ymin=abs(y-round(height/2))      
        
        theta=((angle))*(np.pi/180)
        
        #print x
        #print y
        #print width
        #print height
        #print angle
        #print "gap"
        
        #x1=np.zeros((2,1))
        #x2=np.zeros((2,1))
        #x3=np.zeros((2,1))
        #x4=np.zeros((2,1))
        x1,y1=self.rotate((x,y),(xmin,ymin),theta)
        x2,y2=self.rotate((x,y),(xmax,ymin),theta)
        x3,y3=self.rotate((x,y),(xmax,ymax),theta)
        x4,y4=self.rotate((x,y),(xmin,ymax),theta)
        
        #print x1
        #print y1
        #print x2
        #print y2
        #print x3
        #print y3
        #print x4
        #print y4        
        
        s=[]
        # s.append(vot.Point(x1,y1))
        # s.append(vot.Point(x2,y2))
        # s.append(vot.Point(x3,y3))
        # s.append(vot.Point(x4,y4))

        s.append(Point(x1,y1))
        s.append(Point(x2,y2))
        s.append(Point(x3,y3))
        s.append(Point(x4,y4))

        return Polygon(s), ((x,y), width, height, theta)
    
    
    def PSO_shift(self,image,itr, forests, classifier, fixed_shape):
        
        self._set=0
        Vbest=np.zeros((5,self._popSize))
        for i in tqdm(range(1,self._maxItr+1)):
            if i==1:
                if itr==1:
                    self.popx(self._popSize)
                else:
                    aa=self._popSize-2
                    self._X=np.zeros((5,aa))
                    self.popx(aa)
                    bb=self._prev_P[:,0:self._popSize-aa]
                    
                    XX=np.concatenate((self._X,bb),axis=1)
                    self._X=XX
                    
                self._X=self.check(self._X)    
                self.calcSM(image, forests, classifier, fixed_shape)
                T=np.concatenate((self._Z, self._X), axis=0)
                T=self.pksort(T)
                
                
                pbestval=T[0,:]
                
                pbest=T[1:6,:]
                Xout=pbest
            else:
                
                for j in range(self._popSize):
                    if pbestval[j] <= Z[j]:
                        pbest[:,j]=P[:,j]
                        pbestval[j]=Z[j]
                        Vbest[:,j]=Vnew[:,j]
                Xout=P
            
            # pso = (Xout, pbestval, pbest, Vbest)


            gbestval=np.amax(pbestval)
            arg=np.argmax(pbestval)
            gbest=pbest[:,arg]
            Vnew=self.updateVelocity(Xout,Vbest,pbest,gbest)
            P1=self.updatePosition(Xout,Vnew)
            #P2=self.DEcomp(image,Xcard)
            #P=np.concatenate((P1,P2),axis=1)
            #CCV=np.zeros((5,self._discard))
            #V=np.concatenate((Vnew,CCV),axis=1)
            
            P1=self.check(P1)
            self._X=P1
            
            self.calcSM(image, forests, classifier, fixed_shape)
            Pnet=self._Z
            C=np.concatenate((Pnet,P1,Vnew),axis=0)
            C=self.pksort(C)
            Z=C[0,:]       
            Cn=C[1:11,:]
            P=Cn[0:5,:]
            
            Vnew=Cn[5:10,:]  
            #print Vnew.shape
            
        
        # pso = P

        # pickle_out = open("pso.pickle", "wb")
        # pickle.dump(pso, pickle_out, protocol=2)
        # pickle_out.close()        
        
        self._prev_P=P
        
        fbest=P[:,0]
        self._curr_centroid_x=fbest[0]
        self._curr_centroid_y=fbest[1]
        self._curr_width=fbest[2]
        self._curr_height=fbest[3]
        self._curr_angle=fbest[4] 
        outfinal=self.generateOutput()
        return outfinal
        
    
# # handle = vot.VOT("polygon")
# # selection = handle.region()
# frames_dir = '../new_frames/bag/'
# img_names_list = list()
# for img_name in os.listdir(frames_dir):
#     if('jpg' or 'png' in img_name):
#         img_names_list.append(img_name)
# img_names_list.sort()
# image = cv2.imread(os.path.join(frames_dir, img_names_list[0]))
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


 
# refPt = []
# def click(event, x, y, flags, param):
# 	# grab references to the global variables
#     global refPt
#     if event == cv2.EVENT_LBUTTONDOWN:
#         refPt.append(Point(x, y))
#         cv2.circle(image, (x,y), 1, (0,255,0))

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click)
# while True:
# 	# display the image and wait for a keypress
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
 
# 	# if the 'c' key is pressed, break from the loop
# 	if key == ord("c"):
# 		break

# selection = Polygon(refPt)

# tracker = PKTracker(image, selection)
# t1=0
# t2=0
# for p in selection.points:
#     t1+=p.x
#     t2+=p.y
# t1=int(t1*0.25)
# t2=int(t2*0.25)
# tracker.trans_x=np.array([[t1-30,t1+30]])
# tracker.trans_y=np.array([[t2-30,t2+30]])
# #self.trans_x=np.array([[5,cc-5]])
# #self.trans_y=np.array([[5,rr-5]])
# # itr=0

# def draw_save_img(img, rect, img_name):
#     pt1, pt2, pt3, pt4 = rect.points[0], rect.points[1], rect.points[2], rect.points[3]
#     cv2.line(img, pt1, pt2, [0,255,0], 1)
#     cv2.line(img, pt2, pt3, [0,255,0], 1)
#     cv2.line(img, pt3, pt4, [0,255,0], 1)
#     cv2.line(img, pt4, pt1, [0,255,0], 1)
#     cv2.imwrite(os.path.join(output_dir, img_name), img)


# # output_dir = '../output_PSO2'
# # while True:
# for i, img_name in tqdm(enumerate(img_names_list)):
#     if(i==0):
#         continue
#     # itr=itr+1
#     frame = cv2.imread(os.path.join(frames_dir,img_name), cv2.IMREAD_COLOR)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # imagefile = handle.frame()
#     # if not imagefile:
#         # break
#     # image = cv2.imread(imagefile)
#     reg = tracker.PSO_shift(frame,i)
#     draw_save_img(frame, reg, img_name)
#     # print(reg)
#     # handle.report(reg)

