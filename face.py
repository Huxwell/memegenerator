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

# local modules
from video import create_capture
from common import clock, draw_str

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    #print(rects)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    #print(rects)
    #wait = input("PRESS ENTER TO CONTINUE.")
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
def rect_to_rect(img1,img2,rect1,rect2):
    roi = img1[rect1[1]:rect1[3], rect1[0]:rect1[2]]
    roi2 = img2[rect2[1]:rect2[3], rect2[0]:rect2[2]]
    roi = cv2.resize(roi2,(roi.shape[0],roi.shape[1]))
    img1[rect1[1]:rect1[3], rect1[0]:rect1[2]]=roi
    #cv2.imshow('roi', roi)
    #cv2.imshow('roi2', roi2)
    #cv2.imshow('merged', img1)
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
    meme = cv2.imread("memes/spock.jpg")
    meme_rects = detect(meme, cascade)

    #draw_rects(meme, meme_rects, (0, 255, 0))
    #cv2.imshow('meme', meme)
    rects = []
    img=[]
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        if len(rects)==0:
            time.sleep(.01)
            continue
        print(rects)

        vis = img.copy()
        #draw_rects(vis, rects, (0, 255, 0))

        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t
        vis = rect_to_rect(meme,vis,meme_rects[0],rects[0])
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)





        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()