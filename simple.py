import cv2

# Create a black image
img = cv2.imread('C:\Users\81804\Desktop\Final Project')  # specify the path to an image file
cv2.imshow('Test Window', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
