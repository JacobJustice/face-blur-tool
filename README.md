# Automated Face Blurring/Anonymization
Detects faces contained in an image and then blurs them. Inspired by the protests 

# How to use
-i / --image        path to image you want to blur faces in

-c / --confidence   float from 0-1 describing the minimum confidence required to label a face (lower will mean more faces but more false positives, higher will mean less faces but less false positives)

`python3 blur.py -i <image_name> [-c <0.0 - 1.0>]`

It will display the image pre-blur using matplotlib and after the blurring is added, then save the image to the outputs directory as output.png (overriding if there is an image already there) 

# Dependencies
 + opencv-python
 + matplotlib (not actually necessary so you could remove the parts that it uses if you want)
 + PIL
 + numpy

# Other
Feel free to make pull requests or use this for your own projects.

thanks to Adrian Rosebrock for a tutorial that I modified:
https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/

I don't own any of the images in the repo and will happily delete them if the owner wishes, I used them for demonstration purposes only.

