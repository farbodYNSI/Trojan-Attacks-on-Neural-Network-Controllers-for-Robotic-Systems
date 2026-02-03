import math
import numpy as np
import cv2

class world():

    def __init__(self, wheel_length):
        self.x = 0
        self.y = 0
        self.theta = 0 
        self.left_speed = 0
        self.right_speed = 0
        self.speed = 0
        self.length = wheel_length
        self.background_pic = np.zeros((800,800,3),dtype=np.uint8)
        self.robot_pic = cv2.imread('nav.png')
        self.robot_pic= cv2.resize(self.robot_pic,(self.robot_pic.shape[1]//8,self.robot_pic.shape[0]//8))
        self.originY = self.background_pic.shape[0]//2
        self.originX = self.background_pic.shape[1]//2
        self.path_trajectory = [(self.x,self.y)]
    
    def tick(self, left_speed, right_speed, time = 0.2):
        self.right_speed = right_speed
        self.left_speed = left_speed
        self.speed = (self.right_speed+self.left_speed)/2
        self.theta += (self.right_speed-self.left_speed)/(self.length) * time
        while self.theta > math.pi:
            self.theta -= 2*math.pi
        while self.theta < -math.pi:
            self.theta += 2*math.pi
        self.x += self.speed * time * math.cos(self.theta)
        self.y += self.speed * time * math.sin(self.theta)
        self.path_trajectory.append((self.x,self.y))

    def pos(self):
        return (self.x, self.y , self.theta)
    
    def rotate_image(self,image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        robot_mask = cv2.inRange(image.copy(),np.array([215,150,70]),np.array([255,200,130]))
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        robot_mask = cv2.warpAffine(robot_mask, rot_mat, robot_mask.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result, robot_mask

    def visualize(self):
        robot , robot_mask = self.rotate_image(self.robot_pic.copy(),self.theta*180/math.pi)
        robot[robot_mask==0] = [0,0,0]
        background = self.background_pic.copy()
        background[self.originX-int(self.y)-robot.shape[1]//2:self.originX-int(self.y)+robot.shape[1]//2,self.originY+int(self.x)-robot.shape[1]//2:self.originY+int(self.x)+robot.shape[1]//2,:] = robot
        cv2.line(background,(0,self.originY),(background.shape[1],self.originY),(255,255,255),1)
        cv2.line(background,(self.originX,0),(self.originX,background.shape[0]),(255,255,255),1)
        cv2.putText(background,'X: {:.2f} Y: {:.2f} Theta: {:.2f}'.format(self.x,self.y,self.theta*180/math.pi),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        return background

    def draw_target(self, targets, background):
        for target in targets:
            cv2.circle(background,(self.originY+int(target[0]),self.originX-int(target[1])),2,(0,0,255),-1)
        return background
    
    def draw_path(self, background):
        for i in range(1,len(self.path_trajectory)):
            cv2.line(background,(self.originY+int(self.path_trajectory[i-1][0]),self.originX-int(self.path_trajectory[i-1][1])),(self.originY+int(self.path_trajectory[i][0]),self.originX-int(self.path_trajectory[i][1])),(0,255,0),1)
        return background
    

