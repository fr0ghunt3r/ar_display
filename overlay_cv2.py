# Overlaying HUD with OpenCV functions
import cv2
import numpy as np
import yaml
from optparse import OptionParser
from image_loader import CV2Loader
from objloader_simple import *


class map_class(object):
    def __init__(self, map_pgm, map_yaml, resize: float = 0.35):
        self.img = map_pgm
        h, w, c  = map_pgm.shape
        self.h = h
        self.w = w
        self.c = c
        with open(map_yaml) as f:
            self.map_info = yaml.load(f, Loader=yaml.FullLoader)
        if not ((resize >= 1) and (resize < 0)):
            self.resize(resize)

    def resize(self, resize):
        self.img = cv2.resize(self.img, dsize =(0,0), fx = resize, fy = resize,
                                                      interpolation = cv2.INTER_AREA)
        self.map_info['resolution'] /= resize
        self.map_info['origin'] = [ vec * resize for vec in self.map_info['origin']]

    @classmethod
    def get_pose(cls, vector, slam_map):
        pose = ( int((vector[0] - slam_map.map_info['origin'][0])
                                                                    / slam_map.map_info['resolution']),
                 int(-1 * (((vector[1] - slam_map.map_info['origin'][1]) 
                                                     / slam_map.map_info['resolution']) - slam_map.img.shape[0])))
        return pose

class overlay(object):
    def __init__(self, slam_map):
        self.panel_res   = (100, 100, 3)
        self.panel_start = (10, 200)
        self.print_variables = ("x", "y", "z", "x", "y", "z","w", "ts")
        self.slam_map = slam_map
        # for 3D arrow model
        self.obj = OBJ("./resources/arrow.obj", swapyz=True)
        #self.projection = perspective
        # map origin
        self.create_mask_arrow()
        self.create_panel(self.panel_res)

    def __call__(self, image, pose, timestamp):
        self.image = image
        self.pose = self.parse(pose)
        self.timestamp = timestamp
        self.overlay()
        return self.image

    def parse(self, string):
        pose = [float(var) for var in string.replace(" ", "").split(",")[2:]]
        return pose
        
    def create_panel(self, res):
        self.panel = np.full(res, 255.0)
       
    def create_mask_arrow(self):
        # for 2D arrow image
        # assign self.arrow, self.arrow_h, self.arrow_w,self.mask, self.mask_inv
        self.arrow = cv2.resize(cv2.imread("./resources/arrow.png"), dsize=(128, 120), 
                                                                          interpolation = cv2.INTER_AREA)
        self.warp_arrow()
        self.arrow_h, self.arrow_w, _ = self.arrow.shape
        arrow_gray = cv2.cvtColor(self.arrow, cv2.COLOR_BGR2GRAY)
        ret, self.mask = cv2.threshold(arrow_gray, 10, 255, cv2.THRESH_BINARY)
        self.mask_inv = cv2.bitwise_not(self.mask)

    def warp_arrow(self):
        ref1 = np.array([[0, 0], [128, 0], [128, 120], [0, 120]], dtype = np.float32)
        tgt1 = np.array([[32, 48], [88, 48], [128, 104], [0, 104]], dtype = np.float32)
        perspective = cv2.getPerspectiveTransform(ref1, tgt1)
        self.arrow = cv2.warpPerspective(self.arrow, perspective, (128, 120))

    def draw_arrow(self):
        roi = self.image[120: 120+self.arrow_h, 106:106+self.arrow_w,:]
        img_bg = cv2.bitwise_and(roi, roi, mask = self.mask_inv)
        arrow_fg = cv2.bitwise_and(self.arrow, self.arrow, mask = self.mask)
        dst = cv2.add(img_bg, arrow_fg)
        self.image[120:120+self.arrow_h, 106:106+self.arrow_w] = dst

    def draw_arrow_3d(self, scale3d):
        vertices = self.obj.vertices
        scale_matrix = np.eye(3) * scale3d
        
        for face in self.obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            #points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv2.projectPoints(points.reshape(-1, 1, 3)[0][0],(0,0,0),(0,0,0), () , np.eye(3))
            framePts = np.int32(dst)
            cv2.fillConvexPoly(self.image, framePts, (137, 27, 211))

    def draw_pose(self):
        pass

    def update_map_overlay(self):
        # real-time update of robot location on map 
        resized_pose = map_class.get_pose(self.pose, self.slam_map) 
        circled_img = cv2.circle(self.slam_map.img, (resized_pose[0], resized_pose[1]), 1, (0, 0, 255), -1)
        # update map overlay from given pose
        self.image[ 0 : self.slam_map.img.shape[0], 
                    self.image.shape[1] - self.slam_map.img.shape[1] :, :]  = circled_img

    def update_text(self):
        panel = self.panel.copy()
        for idx, var in enumerate(self.pose):
            text = f"{self.print_variables[idx]}: {self.pose[idx]} "
            cv2.putText(panel, text, (5, 10+(10*idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{self.print_variables[7]}: {self.timestamp} ", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        return panel 

    def overlay(self):
        self.draw_arrow()
        #self.draw_arrow_3d(10)
        panel = self.update_text()
        self.update_map_overlay()
        overlayed_panel = cv2.addWeighted(
                                          self.image[self.panel_start[0] : self.panel_start[0] + self.panel_res[0],
                                          self.panel_start[1] : self.panel_start[1] + self.panel_res[1], :], 
                                          0.3, 
                                          panel, 
                                          0.7, 
                                          0, 
                                          dtype=cv2.CV_32F)

        self.image[self.panel_start[0] : self.panel_start[0] + self.panel_res[0],
                   self.panel_start[1] : self.panel_start[1] + self.panel_res[1], :] = overlayed_panel 

def create_blank(width, height, rgb_color = (0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image    

def main():

    ###
    panel_right = create_blank(320, 240, (220, 220, 220))
    slam_map_img = cv2.imread("../mac-ros/workspace/data/map.pgm")
    map_yaml_dir = "../mac-ros/workspace/src/map.yaml"
    slam_map = map_class(slam_map_img, map_yaml_dir)
    over_image = overlay(slam_map)
    ###

    with open('../mac-ros/workspace/data/tf.txt') as f: 
        pose_data = f.readlines()
    pose_list = []
    for pose in pose_data:
        if (pose.split(',')[3] == 'odom') and (pose.split(',')[4] == 'base_link'):
            pose_list.append((float(pose.split(',')[5]),float(pose.split(',')[6])))

    with open('../mac-ros/workspace/data/tf_trans.txt') as f: 
        trans_data = f.readlines()
    trans_list = []
    for pose in trans_data:
        comma_separate = pose.split(",")
        comma_separate[0] = comma_separate[0].replace("- Translation: [", "")
        comma_separate[2] = comma_separate[2].replace("]", "").replace("\n", "")
        trans_list.append((float(comma_separate[0]),float(comma_separate[1])))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    try:
        images = CV2Loader("../mac-ros/workspace/data")
        for idx, (image, time) in enumerate(images):
            print(f"processing time of #{idx} of image: {time}")
            image = np.hstack((image, panel_right))
            #image = over_image(image, trans_list[int(idx * (len(trans_list)/len(images)))], "1234555")
            image = over_image(image, "map, base_link, 3.71, 6.65, -0.06, 0.03, -0.07, 0.95, -0.29", "1234555")
            cv2.imshow('image', image) 
            cv2.waitKey(15)
    except Exception as e:
        print("Exception:", e)
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
