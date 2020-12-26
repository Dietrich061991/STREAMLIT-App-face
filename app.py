from cv2 import cv2
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np 
import os


def load_image(img):
    imag = Image.OpenCv(img)
    return imag

face_cascade = cv2.CascadeClassifier('free_cognitive/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('free_cognitive/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('free_cognitive/haarcascade_smile.xml')

def detection_faces(our_image):
    new_image = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_image,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detection face
    faces = face_cascade.detectMultiScale(gray,1.1, 4)
    #Draw rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img, faces

def detection_smiles(our_image):
    new_image = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_image,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detection smiles
    smiles = smile_cascade.detectMultiScale(gray,1.1, 4)
    #Draw rectangle around the smiles
    for (x,y,w,h) in smiles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img 

def detection_eyes(our_image):
    new_image = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_image,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detection eyes
    eyes = eye_cascade.detectMultiScale(gray,1.3, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img

def cartonize_image(our_image):
    new_image = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_image,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # edges
    gray = cv2.medianBlur(gray,5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    #color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    #cartoon 
    cartoon = cv2.bitwise_and(color, color,mask=edges)
    return cartoon

def cannize_image(our_image):
    new_image = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_image,1)
    img = cv2.GaussianBlur(img,(11, 11),0)
    canny =cv2.Canny(img,100, 150)
    return canny




def main():
    '''face detection app'''

    st.title('SIMPLE STREAMLIT DASHBOARD TO FACE DETECTION APP')
    st.header('Build white streamlit and OpenCv')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity',activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
    
   

        image_file = st.file_uploader('Upload Image', type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            #st.write(type(our_image))
            st.image(our_image)

        enhance_type =st.sidebar.radio('Enhance Type',['Original','Gray-scale','Contrast','Brightness','Blurring'])
        
        if enhance_type == 'Gray-scale':
            new_imag = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_imag,1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #st.write(new_imag)
            st.image(gray)

        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast", 0.5 , 3.5)
            enhancer = ImageEnhance.Contrast (our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness", 0.5 , 3.5)
            enhancer = ImageEnhance.Brightness (our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Blurring':
            blur_rate = st.sidebar.slider("Blurring", 0.5 , 3.5)
            new_imag = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_imag,1)
            blur_imag = cv2.GaussianBlur(img, (11 , 11), blur_rate)
            st.image(blur_imag)

        else:
            st.image(our_image,width=300)

        #face detection
        taske =['Faces','Smiles','Eyes','Cannize','Cartonize']
        feature_choice = st.sidebar.selectbox('Find Feature', taske)
        
        if st.button('Process'):

            if feature_choice == 'Faces':
                result_imag, result_faces = detection_faces(our_image)
                st.image(result_imag)
                st.success('Found {} faces'.format(len (result_faces)))

            elif feature_choice == 'Smiles':
                result_imag = detection_smiles(our_image)
                st.image(result_imag)
                

            elif feature_choice == 'Eyes':
                result_imag = detection_eyes(our_image)
                st.image(result_imag)
                

            elif feature_choice == 'Cartonize':
                result_imag = cartonize_image(our_image)
                st.image(result_imag)
                

            elif feature_choice == 'Cannize':
                result_cann = cannize_image(our_image)
                st.image(result_cann)
                


    elif choice =='About':
        st.subheader('Author: Sedami Dietrich Montcho')
        st.subheader('Data science')
        st.text('Contact : +55 83996049106')
        st.text('Email : didimontcho@gmail.com')
        st.text('Country: Brazil')
        st.success('Never Give Up')

        

            

if __name__ == "__main__":
             main()



