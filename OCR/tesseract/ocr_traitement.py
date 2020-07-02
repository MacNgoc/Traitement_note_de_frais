import pytesseract
from pytesseract import image_to_string
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def list_files(directory):
    """
    this function is to list all files .jpg, jpeg, png in a directory
    """
    files=[]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"): 
            files.append(os.path.join(directory, filename))
            continue
        else:
            continue
    return files



def apply_threshold(img, argument):    
    switcher = {        
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],        
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        18: cv2.adaptiveThreshold(cv2.medianBlur(img, 7), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),        
        19: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),        
        20: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)    
    }    
    return switcher.get(argument, "Invalid method")



def traitement(path_to_image):
    img = cv2.imread(path_to_image)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Convert to gray    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Apply dilation and erosion to remove some noise    
    kernel = np.ones((1, 1), np.uint8)  # we can choose kernel size = (3,3), (5, 5), (7,7)  
    img = cv2.dilate(img, kernel, iterations=1)    
    img = cv2.erode(img, kernel, iterations=1)
    

    # Apply threshold to get image with only black and white    
    img = apply_threshold(img, 3)
    
    # reduce some noise at border
    img[0:10,:] = 255
    img[:, 0:10] = 255
    img[img.shape[0]-10:img.shape[0], :]=255
    
    #imshow image
    #plt.imshow(img,'gray')
    
    return img


def traitement_bilateralFilter(path_to_image):
    img = cv2.imread(path_to_image)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    img = cv2.bilateralFilter(img,9,75,75)
    img = cv2.multiply(img, 1.2)
    
    return img
    




def ocr_tesseract(path_to_image, type_save, type_traitement):
    """
    type_traitement = {1: traitement avec dilate et erod,2:traitement bilateralFilter , else: nothing}
    type_save = {1: traitement avec dilate et erod, 2: traitement bilateralFilter, else: origin }
    """
    

    if type_traitement==1:
        img = traitement(path_to_image)
    elif type_traitement == 2:
        img = traitement_bilateralFilter(path_to_image)
    else:
        img = cv2.imread(path_to_image)


    
    # OCR
    # mode engin 1 et lang Fra et mode segmentation 6
    custom_config = r'--oem 1 -l fra --psm 6'
    text_fra_6 = '*FRA6*: ' +  image_to_string(img, config=custom_config) + '\n'

    custom_config = r'--oem 1 -l fra --psm 7'
    text_fra_7 = '*FRA7*: ' +  image_to_string(img, config=custom_config) + '\n'
    
    
    # mode engin 1 et lang eng et mode segmentation 7
    custom_config = r'--oem 1 -l eng --psm 6'
    text_eng_6 = '*ENG6*: '+  image_to_string(img, config=custom_config) + '\n'

    custom_config = r'--oem 1 -l eng --psm 7'
    text_eng_7 = '*ENG7*: '+  image_to_string(img, config=custom_config) + '\n'
    #text_eng_orig = 'ENG7: ' + image_to_string(img_origin, config=custom_config) + '\n'
    
    # mode engin 1 et mode segmentaiton 6 et whitelist
    custom_config = r' --oem 1 -c tessedit_char_whitelist=0123456789,/%. --psm 6'
    text_number = '*NUM6*: '  +  image_to_string(img, config=custom_config) 
    #text_number_orig = 'NUM6: ' + image_to_string(img_origin, config=custom_config)
    
    
    image_name =os.path.splitext(os.path.basename(path_to_image))[0]
    
    fig =plt.figure(figsize=(25, 25))
    
    text = text_fra_6 + text_fra_7 + text_eng_6 + text_eng_7 + text_number
    
    
    ax= fig.add_subplot()
    plt.imshow(img, 'gray')
    ax.set_title(text, fontsize = 25)
    #plt.show()
    #plt.axis('off')
    
    #save image
    """
    switcher = {
    1: fig.savefig('images/{}_origin.jpg'.format(image_name), bbox_inches='tight'),
    2: fig.savefig('images/{}_1.jpg'.format(image_name), bbox_inches='tight'),
    3: fig.savefig('images/{}_2.jpg'.format(image_name), bbox_inches='tight')
    }
    """

    if type_save == 1:
        fig.savefig('images/{}_1.jpg'.format(image_name), bbox_inches='tight')
    elif type_save == 2:
        fig.savefig('images/{}_2.jpg'.format(image_name), bbox_inches='tight')
    else:
        fig.savefig('images/{}_origin.jpg'.format(image_name), bbox_inches='tight')
    print("Image saved successfully to folder /images/")
   

    

def main(list_files_images):
    for file in list_files_images:
        ocr_tesseract(file, 3, 3)


directory = r"C:\Users\thibichngoc.mac\Documents\MAC_NGOC\Projet\DeepLearning\test_detection\couper"
list_path_images = list_files(directory)
main(list_path_images)   
    