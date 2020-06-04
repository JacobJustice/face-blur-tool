# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

"""
detect_faces_cascade: a simple face detection using multiple Haar Cascade models built in to Open CV
    @image_gray: input image; Must be grayscale

    return: a list of x, y, width, and height tuples; returns rectangles around faces in the image
	[ (x1,y1,w1,h1), ... ]
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
thanks to Adrian Rosebrock for the starter code and tutorial (link found in the readme)

detect_faces_dnn: uses the dnn built into opencv
    @image: input image
    @confidence: *OPTIONAL* 

    @return:  a list of x, y, width, and height tuples; returns rectangles around faces in the image
	[ (x1,y1,w1,h1), ... ]

"""

model_fn = './data/dnn/res10_300x300_ssd_iter_140000.caffemodel'
proto_fn = './data/dnn/deploy.prototxt.txt'
def detect_faces_dnn(image, confidence=0.150):
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto_fn, model_fn)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()


    # make empty face_rects
    face_rects = [] 

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        prediction_confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if prediction_confidence > confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
     
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(prediction_confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

#            cv2.rectangle(image, (startX, startY), (endX, endY),
#                (0, 0, 255), 2)
#            cv2.putText(image, text, (startX, y),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            face_rects.append((startX, startY, endX-startX, endY-startY))

    # show the output image

    return face_rects


"""
main: 	if the file is called, it expects an image as a parameter
	 it runs that image through detect_faces_cascade and then displays the image with green boxes around the faces

	used for testing not intended to be called by other functions or programs
"""
def main():

    # configure ArgumentParser
    ap = argparse.ArgumentParser()
    	# image is a required argument 
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image")

    args = vars(ap.parse_args())

    # load image
    image = cv2.imread(args['image'])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_rects = detect_faces_dnn(image)
    for (x,y,w,h) in faces_rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(convertToRGB(image))
    plt.show()

if __name__ == "__main__":
    print("hello world")
    main()
