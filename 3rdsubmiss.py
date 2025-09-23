import cv2
import matplotlib.pyplot as plt
import numpy as np

# original video
cap = cv2.VideoCapture('/Users/suryasaikadali/Downloads/open_Cv/CVIP/M5/videos/greenscreen-demo.mp4')
# new background image
background = cv2.imread('/Users/suryasaikadali/Downloads/open_Cv/CVIP/M5/images/dino-reichmuth-kk3W5-0b6e0-unsplash.jpg', 1)

# it is important to read the first frame to get the frame dimensions
ret, frame = cap.read()

color_list = []

# function to compute lower and upper bounds for selected colors
'''this function is capable of storing pixel values of multiple clicks(color values) and 
   returns the min and max for each channel and we can also adjust the tolerance using trackbar'''
def bounds():
    global color_list
    if not color_list:
        return None, None
    
    lower_bounds = np.min(color_list, axis = 0).tolist()
    upper_bounds = np.max(color_list, axis = 0).tolist()

    return np.array(lower_bounds), np.array(upper_bounds)

# trackbar callback function (does nothing)
def toler(*args):
    pass

# mouse callback function to select color patches
def colorpatchselector(event,x,y,flags,param):
    global frame, color_list
    if event == cv2.EVENT_LBUTTONDOWN:
        colors = frame[y,x].astype(int)
        color_list.append(colors)

        # select a rectangular patch around the clicked point and compute its average color
        patch = frame[max(y-350,0):min(y+350, frame.shape[0]-1), 
                      max(x-350, 0):min(x+350, frame.shape[1]-1)].mean(axis = (0,1)).astype(int)
        color_list.append(patch)
        print("average color of the patch is:", patch)
        print("selected colors are:", color_list)

        #update bounds
        lower_bounds, upper_bounds = bounds()
        print(lower_bounds, upper_bounds)
       

# create a window
cv2.namedWindow('frame')

# set mouse callback function
cv2.setMouseCallback("frame", colorpatchselector)

# create a trackbar for tolerance
cv2.createTrackbar('tolerance', 'frame', 0, 150, toler)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display_frame = frame.copy()

    background = cv2.resize(background,(frame.shape[1], frame.shape[0]))
    if color_list:
        lower_bounds, upper_bounds = bounds()
        tol = cv2.getTrackbarPos('tolerance', 'frame')

        # compute lower and upper bounds with tolerance
        lower = np.clip((lower_bounds - tol),0, 255)
        upper = np.clip((upper_bounds + tol), 0, 255)

        # create a mask based on the color range
        mask = cv2.inRange(frame, lower, upper)
        # invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # extract foreground and background using the masks
        foreground = cv2.bitwise_and(frame, frame, mask = mask_inv)
        back = cv2.bitwise_and(background, background, mask = mask)

        # merging foreground and background
        display_frame = cv2.add(foreground, back)

    cv2.imshow('frame', display_frame)
    k = cv2.waitKey(3) & 0XFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()