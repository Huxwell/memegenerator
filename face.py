#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import time
import sys
print(sys.version)

# local modules
from video import create_capture
from common import clock, draw_str

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
  
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]

    return rects

def nothing(x):
    pass

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
def rect_to_rect(img1,img2,rect1,rect2):
    roi = img1[rect1[1]:rect1[3], rect1[0]:rect1[2]]
    roi2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2]]
    roi = cv2.resize(roi2,(roi.shape[0],roi.shape[1]))
    img1[rect1[1]:rect1[3], rect1[0]:rect1[2]]=roi
    return img1
if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    meme = cv2.imread("memes/meme0.jpg")
    meme_rects = detect(meme, cascade)
    track = 0

    cv2.namedWindow('facedetect')
    cv2.createTrackbar("meme",'facedetect',0,7,nothing)

    rects = []
    img=[]
    while True:
        new_track = cv2.getTrackbarPos('meme','facedetect')
        if(track != new_track):
            print(new_track)
            track = new_track
            meme = cv2.imread("memes/meme"+str(track)+".jpg")
            meme_rects = detect(meme, cascade)

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        if len(rects)==0:
            time.sleep(.01)
            continue

        vis = img.copy()

        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
           
                if len(subrects) == 2:
                    y = int((subrects[0][1] + subrects[1][1]) / 2)

                    '''s_img = cv2.imread("memes/small_glasses.png")
                    x_offset=y_offset=0
                    vis_roi[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img'''
                    x_offset=int(vis_roi.shape[0]*0.1)
                    y_offset=y
                    s_img = cv2.imread("memes/small_glasses.png",-1)
                    s_img = cv2.resize(s_img,(int(vis_roi.shape[1]*0.8),int(vis_roi.shape[1]*0.38*0.8)))

                    print(s_img.shape)
                    for c in range(0,3):
                        vis_roi[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] =\
                        s_img[:,:,c] * (s_img[:,:,3]/255.0) +  vis_roi[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1], c] * (1.0 - s_img[:,:,3]/255.0)



        dt = clock() - t
        vis = rect_to_rect(meme,vis,meme_rects[0],rects[0])
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)




        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
