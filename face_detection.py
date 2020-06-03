# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

"""
detect_faces_cascade: a simple face detection using multiple Haar Cascade models built in to Open CV
    @image_gray: input image; Must be grayscale

    return: a list of x, y, width, and height tuples; returns rectangles around faces in the image
	return [ (x1,y1,w1,h1), ... ]
"""
def detect_faces_cascade(image_gray):
    haar_cascade_frontface = cv2.CascadeClassifier('/home/jacob/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    front_faces_rects = haar_cascade_frontface.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5);
    print("front faces detected: ", len(front_faces_rects))

    haar_cascade_profileface = cv2.CascadeClassifier('/home/jacob/.local/lib/python3.8/site-packages/cv2/data/haarcascade_profileface.xml')
    profile_faces_rects = haar_cascade_profileface.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5);
    print("profile faces detected: ", len(profile_faces_rects))

    # convert to list to make things easier to extend; I'm sure there's a better way
    front_faces_rects = list(front_faces_rects)
    profile_faces_rects = list(profile_faces_rects)
    front_faces_rects.extend(profile_faces_rects)


    return front_faces_rects


"""
main: 	if the file is called, it expects an image as a parameter
	 it runs that image through detect_faces_cascade and then displays the image with green boxes around the faces

	used for testing not intended to be called by other functions or programs
"""
def main():
    def convertToRGB(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # configure ArgumentParser
    ap = argparse.ArgumentParser()
    	# image is a required argument 
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")

    args = vars(ap.parse_args())

    # load image
    image = cv2.imread(args['image'])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_rects = detect_faces_cascade(image_gray)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(convertToRGB(image))
    plt.show()

if __name__ == "__main__":
    print("hello world")
    main()
