import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def show_image(org_img , msk_img):
    # Masking original image
    result_img = cv2.bitwise_and(msk_img,org_img,)
    cv2.imshow("image",result_img) # Showing image
    cv2.waitKey(0) #Wait key apply 
    cv2.destroyAllWindows() # Close all windows

def prosessing_image(input_image):
    img = input_image.copy()
    # gray scale image
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Image binalization
    ret,thresh = cv2.threshold(imgray,110,255,cv2.THRESH_BINARY)
    #District prosess area
    xmin ,xmax = 44, 224
    ymin , ymax = 110 , 216
    roi = (xmin, ymin, xmax, ymax)
    s_roi = thresh[roi[1]: roi[3], roi[0]: roi[2]]

    #make contours
    contours, hierarchy = cv2.findContours(s_roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE,offset=(xmin,ymin))
    # remove small object
    contours = list(filter(lambda x: cv2.contourArea(x) > 24.0 , contours))
    # DEBUG /draw contours to obejects
    ##contours_img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
   
    ## Making mask from Contours
    contour_mask = np.zeros_like(imgray)
    mask_image = cv2.drawContours(contour_mask, contours, -1, color=255, thickness=-1)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR) #Resize to same as original
    
    ''' DEBUGG
    titles = ['Original Image','BINARY','Contours_img','mask']
    images = [input_image, thresh, contours_img,mask_image]

    for i in range(4):
        plt.subplot(1,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()
    '''
    return mask_image




original_image = cv2.imread('milkdrop.bmp',cv2.IMREAD_UNCHANGED)
 # convert RGB to BGR
if original_image.ndim == 3:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
show_image(original_image , prosessing_image(original_image))