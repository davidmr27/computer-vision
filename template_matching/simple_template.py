import argparse
import cv2

#Argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, required=True, help="path to input image")
ap.add_argument('-t', '--template', type=str, required=True, help="path to template image")
args = vars(ap.parse_args())

#Load the input image and template image from disk
print("[INFO] loading images...")
image = cv2.imread(args['image'])
template = cv2.imread(args['template'])

cv2.imshow("Image", image)
cv2.imshow("template", template)

#Convert to scale gray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# perfomr template matching
#Metoodo que funcionaron correctament con template matching
# result = cv2.matchTemplate(image=imageGray,templ=templateGray,method=cv2.TM_CCOEFF_NORMED)
# result = cv2.matchTemplate(image=imageGray,templ=templateGray,method=cv2.TM_CCOEFF)
result = cv2.matchTemplate(image=imageGray,templ=templateGray,method=cv2.TM_CCORR_NORMED)
(minVal,maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result) #busca los valores minimos y maxinos en un array
print(f"[INFO]: {minVal}, {maxVal}, {minLoc}, {maxLoc}")

(startX, startY) = maxLoc
endX = startX + template.shape[1]
endY = startY + template.shape[0]

#Detect the bounding box on the image
cv2.rectangle(image, (startX,startY), (endX, endY), (255,0,0), 2)

cv2.imshow("Output", image)
cv2.waitKey(0)