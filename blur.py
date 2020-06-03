import cv2
import numpy as np
from PIL import Image

"""
copied from 
https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

author: Adrian Rosebrock

blurs whole image and returns blurred image
"""
def blur(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
    	kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
    	kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)

"""
blur_faces: blurs every face in image

    @face_rects: a list of x, y, width, and height tuples; returns rectangles around faces in the image

    @return: numpy image

"""
def blur_faces(image, face_rects, factor=3.0):
    pil_image = Image.fromarray(image)

        # for every set of face coordinates
    for (x,y,w,h) in face_rects:
            # isolate the face in the image
        face_image = image[y:y+h, x:x+w]

            # blur the isolated image
        face_image = blur(face_image, factor)

#        plt.imshow(convertToRGB(face_image))
#        plt.show()

	# may be slow because of PIL conversion not really sure
        pil_image.paste(Image.fromarray(face_image), (x,y))

    return np.asarray(pil_image)

def main():
    import matplotlib.pyplot as plt
    import argparse
    from face_detection import detect_faces_cascade
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
    image = blur_faces(image, faces_rects)

    plt.imshow(convertToRGB(image))
    plt.show()


if __name__ == "__main__":
    main()
