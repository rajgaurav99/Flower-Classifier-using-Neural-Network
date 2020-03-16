# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:52:26 2020

@author: rajga
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from PIL import ImageTk
import cv2
from tkinter import filedialog, Button, PhotoImage, Label, E, W, Tk, messagebox, LabelFrame
model = load_model("flower.h5")






img_flag=False
def load_image():
    if(img_flag):
        img = image.load_img(img_input, target_size=(150, 150))
        img_tensor = image.img_to_array(img)                    
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        pred = model.predict_classes(img_tensor)[0]
        if(pred==0):
            result='Daisy'
        elif(pred==1):
            result='Dandellion'
        elif(pred==2):
            result='Rose'
        elif(pred==3):
            result='Sunflower'
        elif(pred==4):
            result='Tulip'
        else:
            result='Could not classify'
        result_text.configure(text=result)
    else:
        messagebox.showinfo("Error", "Select a valid image")

def select_image():
    global img_panel
    global img_input
    img_input=filedialog.askopenfilename(title = "Select Flower Picture",filetypes=[("Image files (JPEG)","*.jpg")])
    if(len(img_input)>0):
        image = cv2.imread(img_input)
        image=cv2.resize(image,(240,240))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        img_panel.configure(image=image)
        img_panel.image=image
        global img_flag
        img_flag=True
        

root=Tk()
root.title("Flower Classifier")
root.geometry("600x300")
root.resizable(False,False)
root.configure(bg="#e3f4fc")


label1=Label(root,text="Flower Classifier",font='Helvetica 12 bold', bg='#e3f4fc')
label1.grid(row=0,column=0,sticky=E+W,padx=50)

button1=Button(root, text="Select Flower Image file",command=select_image,width=30)
button1.grid(row=1,column=0,sticky=E,padx=50)

button2=Button(root, text="Analyze Flower Image",command=load_image,width=30)
button2.grid(row=2,column=0,sticky=E,padx=50)

img_frame=LabelFrame(root,text="Flower Image")
img_frame.grid(row=0, column=1, columnspan=2, rowspan=4,pady=10,sticky=E)

default_image=PhotoImage(file="default.png")
img_panel = Label(img_frame,image=default_image)
img_panel.image=default_image
img_panel.grid()

result_frame=LabelFrame(root,text="Result")
result_frame.grid(row=3,column=0,sticky=E,padx=50)
result_text=Label(result_frame,text="",width=30)
result_text.grid()
root.mainloop()
