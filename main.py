import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def findWleft(row, image):
    for edge in range(6,-1,-1):
        cropped_section = image[edge*100 - 100:edge*100, row*100:row*100 + 100]
        threshold = 200
        binary_image = np.where(cropped_section > threshold, 255, 0)
        white_pixels = np.sum(binary_image == 255)
        if(white_pixels>300):
            image_copy = image.copy()
            cv2.rectangle(image_copy, (edge*100 - 100, row*100), (edge*100, row*100 + 100), (0, 255, 0), 2)
            cv2.imshow('Cropped Section', image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return (edge*100, row*100)
    return (edge*100, row*100)


def identify_heart(image):
    margins=[]
    for row in range(6):
        for edge in range(6):
            cropped_section = image[edge*100:edge*100+100, row*100:row*100+100]
            threshold = 200
            binary_image = np.where(cropped_section > threshold, 255, 0)
            white_pixels = np.sum(binary_image == 255)
            print(white_pixels)
            if white_pixels>40:
                break    
        margins.append((edge*100, row*100))
    return margins


def left_ventricle(image, margin):
    x1, y1 = margin
    row = y1
    for num in range(x1,600,100):
        cropped_section = image[num*100:num*100+100, row:row+100]
        binary_image = np.where(cropped_section > 100, 255, 0)
        white_pixels = np.sum(binary_image == 255)
        if white_pixels<300:
            cv2.putText(image, "LV", (num+200,row+150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            break

def right_ventricle(image, margin):
    x1, y1 = margin
    row = y1
    for num in range(x1,600,100):
        cropped_section = image[num*100:num*100+100, row:row+100]
        binary_image = np.where(cropped_section > 100, 255, 0)
        white_pixels = np.sum(binary_image == 255)
        if white_pixels<300:
            cv2.putText(image, "RV", (num+50,row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            break

def left_atrium(image, margin):
    x1, y1 = margin
    row = y1
    for num in range(x1,600,100):
        cropped_section = image[num*100:num*100+100, row:row+100]
        binary_image = np.where(cropped_section > 100, 255, 0)
        white_pixels = np.sum(binary_image == 255)
        if white_pixels<300:
            cv2.putText(image, "LA", (num+200,row+100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            break

def right_atrium(image, margin):
    x1, y1 = margin
    row = y1
    for num in range(x1,600,100):
        cropped_section = image[num*100:num*100+100, row:row+100]
        binary_image = np.where(cropped_section > 100, 255, 0)
        white_pixels = np.sum(binary_image == 255)
        if white_pixels<300:
            cv2.putText(image, "RA", (num+100,row+100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            break


#read image
cardiograph_og = cv2.imread('cardiograph2.jpg')
cardiograph = cv2.resize(cardiograph_og, (600, 600))

if cardiograph is None:
    print("Error: image not found")
else:
    gray_cardio = cv2.cvtColor(cardiograph, cv2.COLOR_BGR2GRAY)
    blurred_cardio = cv2.GaussianBlur(gray_cardio, (5,5), 0)
    edges = cv2.Canny(blurred_cardio, 10, 50)
    
    #find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = np.zeros_like(cardiograph)
    contoured_image2 = np.zeros_like(cardiograph)
    cv2.drawContours(contoured_image, contours, -1, (0, 0, 255), 2)

    cv2.putText(contoured_image, "Left Ventricle", (313,265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    height, width = edges.shape
    for y in range(height):
        for x in range(width):
        #if the pixel value is greater than 0, it's an edge
            if edges[y, x] > 0:
                contoured_image2[y,x] = contoured_image[y,x]


    margins=identify_heart(contoured_image2)
    print(margins) #prints approximate edges of the heart 
    left_ventricle(contoured_image2,margins[1])
    right_ventricle(contoured_image2,margins[3])
    left_atrium(contoured_image2,margins[4])
    right_atrium(contoured_image2,margins[4])
    cv2.imshow('Contoured & Labelled Cardiograph', contoured_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

    
