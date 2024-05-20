import cv2
import numpy as np
import streamlit as st


st.title('Filters application')
# streamlit run Filters_App.py

upload=st.file_uploader("Choose an image",["png","jpg"])
#print(upload)

def blackwhite(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_img


def vignette(img, level=2):
    height, width = img.shape[:2]

    X_kernel = cv2.getGaussianKernel(width, width / level)
    Y_kernel = cv2.getGaussianKernel(height, height / level)

    kernel = Y_kernel * X_kernel.T

    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette


def pencil_sketch(img, k_size):
    blur=cv2.GaussianBlur(img,(k_size,k_size),0,0)
    sketch,_=cv2.pencilSketch(blur)
    return sketch

def HDR(img,sigma_s=10,sigma_r=0.1):
    HD_img = cv2.detailEnhance(img,sigma_s=sigma_s,sigma_r=sigma_r)
    return HD_img

def stylization_fun(img,sigma_s=10,sigma_r=0.1):
    blur=cv2.GaussianBlur(img,(5,5),0,0)
    stylization_img = cv2.stylization(blur,sigma_s=sigma_s,sigma_r=sigma_r)
    return stylization_img

def brightness(img,level):
    bright_img=cv2.convertScaleAbs(img,beta=level)
    return bright_img
if upload is not None:
    raw_bytes=np.asarray(bytearray(upload.read()),dtype=np.uint8) #Encode the image to bytes
    img=cv2.imdecode(raw_bytes,cv2.IMREAD_COLOR) #Decoding the image to be handled using cv2
    input_col, output_col=st.columns(2)
    with input_col:
        st.header("Original Image")
        st.image(img,channels="BGR",use_column_width=True)

    st.header("Filters list")

    options=st.selectbox("Select Filters",("None","Black and White","Vignette","Pencil sketch","HDR","Stylization","Brightness"))


    color="BGR"

    output = img

    if options == "Black and White":
        output=blackwhite(img)
        color="GRAY"
    elif options == "Vignette":
        level = st.slider("level", 0, 5, 1)
        output=vignette(img,level)

    elif options == "Pencil sketch":
        kernel=st.slider("Kernel size",1,9,1,step=2)
        output = pencil_sketch(img,kernel)
        color="GRAY"

    elif options =="HDR":
        output=HDR(img)

    elif options == "Stylization":
        sigma=st.slider("Sigma",0,200,40,step=10)
        output = stylization_fun(img,sigma)

    elif options == "Brightness":
        level=st.slider("Brightness",-50,50,10,step=5)
        output=brightness(img,level)

    else:
        pass

    with output_col:
        st.header("Output image")
        st.image(output,channels=color,use_column_width=True)

