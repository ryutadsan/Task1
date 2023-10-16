import numpy as np
import cv2

def show_image(org_img , msk_img):
    # Compressed image
    org_img = cv2.resize(org_img, None,fx=0.1,fy=0.1, interpolation = cv2.INTER_CUBIC)    
    msk_img = cv2.resize(msk_img, None,fx=0.1,fy=0.1, interpolation = cv2.INTER_CUBIC)
    # Masking original image
    result_img = cv2.bitwise_and(msk_img,org_img,)
    cv2.imshow("image",result_img) # Showing image
    cv2.waitKey(0) #Wait key apply 
    cv2.destroyAllWindows() # Close all windows

def prosessing_image(input_image):
    img = input_image.copy() #original image
    ## Image binalization : 金属パネルが白く、ノイズが多いため色相反転して輪郭を抽出
    ret,thresh = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
    
    #blur
    blur_img = cv2.medianBlur(thresh, ksize=33)
    #additional blur area
    xmin ,xmax = 1920 , 5070
    ymin , ymax = 1230 , 4070
    roi = (xmin, ymin, xmax, ymax)
    s_roi = thresh[roi[1]: roi[3], roi[0]: roi[2]]
    s_roi = cv2.medianBlur(s_roi,ksize=61)
    blur_img[roi[1]: roi[3], roi[0]: roi[2]] = s_roi

    #make contours
    contours, hierarchy = cv2.findContours(blur_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # remove small object
    contours = list(filter(lambda x: cv2.contourArea(x) > 100000.0 , contours))
    
    ## Making mask from Contours
    contour_mask = np.zeros_like(img)
    mask_image = cv2.drawContours(contour_mask, contours, -1, color=255, thickness=-1)
    # Invert : マスク画像を再度色相反転
    ret,mask_inv = cv2.threshold(mask_image,100,255,cv2.THRESH_BINARY_INV)

    #Remove noise : オープニング処理で白色ノイズを除去
    neiborhood = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
    img_dilate = cv2.dilate(mask_inv,neiborhood,iterations=6)
    img_erode = cv2.erode(img_dilate,neiborhood,iterations=16)
    
    return img_erode



#import original image
original_image = cv2.imread('metal_panel.jpg',cv2.IMREAD_UNCHANGED)
prosessed_img = prosessing_image(original_image) #prosessing image
show_image(original_image,prosessed_img) #show image