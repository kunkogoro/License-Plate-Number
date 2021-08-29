
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import font
from tkinter.font import families
from PIL import ImageTk,Image
import os
import glob

from numpy.core import machar
from recognition import E2E
import cv2
from pathlib import Path
import argparse
import time

from numpy.lib.arraypad import pad



class Main(Tk):
    def __init__(self):
        super(Main,self).__init__()
        self.title('Nhận dạng biển số xe')
        # lấy chiều dài màng hình 
        srcW = self.winfo_screenwidth()
        # lấy chiểu cao màng hình
        srcH = self.winfo_screenheight()
        self.geometry("700x500+%d+%d"%(srcW/2-350,srcH/2-250))
        self.resizable(width=False,height=False)


        self.frame_result = LabelFrame(self,text="Kết quả",padx=10,pady=10,width=200,height=500)
        # frame_result.pack(side=LEFT,fill="both",expand="yes")
        self.frame_result.grid(row=0,column=0,sticky=N)
        # button nhấn vào để load hình ảnh

        self.frame_plate = LabelFrame(self.frame_result,text="Biển số",padx=3,pady=3,width=400,font="arial 10 bold",height=50)
        self.frame_plate.grid(row=1,column=0)
        self.label_plate = Label(self.frame_plate,text="Biển số",width=22)
        self.label_plate.pack()


        self.frame_plate_color = LabelFrame(self.frame_result,text="Đổi màu",padx=3,pady=3,width=200,height=100)
        self.frame_plate_color.grid(row=2,column=0)
        self.label_plate_color = Label(self.frame_plate_color,text="Đổi màu",width=22)
        self.label_plate_color.pack()


        self.frame_character = LabelFrame(self.frame_result,text="Kết quả nhận dạng",padx=3,pady=3,width=200,height=100)
        self.frame_character.grid(row=3,column=0)
        self.edit_character = Label(self.frame_character,width=22,text="Chữ nhận dạng")
        self.edit_character.pack()

        self.frame_time = LabelFrame(self.frame_result,text="Model",padx=10,pady=10,width=200,height=100)
        self.frame_time.grid(row=4,column=0)
        self.label_time = ttk.Combobox(self.frame_time,width=22)
        self.label_time["value"] = ("CNN","SVM")
        self.label_time.current(0)
        self.label_time.pack()


        self.btn = Button(self.frame_result,text="Lấy hình",padx=50,pady=7,font="arial 10 bold",command=self.open)
        self.btn.grid(row=5,column=0,pady=7)

        self.btn_processing = Button(self.frame_result,text="Thực hiện",padx=50,pady=7,font="arial 10 bold",command=self.processing)
        self.btn_processing.grid(row=6,column=0,pady=7)

        self.btn_reset = Button(self.frame_result,text="Làm mới",padx=50,pady=7,font="arial 10 bold",command=self.reset)
        self.btn_reset.grid(row=7,column=0,pady=7)

        self.btn_exit = Button(self.frame_result,text="Thoát",padx=50,pady=7,font="arial 10 bold",command=self.close)
        self.btn_exit.grid(row=8,column=0,pady=7)



        self.frame_image = LabelFrame(self,text="Hình ảnh",padx=5,pady=5,width=500,height=500)
        # frame_image.pack(side=RIGHT)
        self.frame_image.grid(row=0,column=1)
        self.label_iamge = Label(self.frame_image,text="Hình ảnh",width=500,height=500)
        self.label_iamge.pack()
        
# win = Tk()

# win.title('Nhận dạng biển số xe')
# # lấy chiều dài màng hình 
# srcW = win.winfo_screenwidth()
# # lấy chiểu cao màng hình
# srcH = win.winfo_screenheight()
# win.geometry("700x500+%d+%d"%(srcW/2-350,srcH/2-250))
# win.resizable(width=False,height=False)


    def close(self):
        main.destroy()


    def reset(self):
        self.label_plate.config(image="",text="Biển số",width=22)
        self.label_plate_color.config(image="",text="Đổi màu",width=22)
        self.edit_character.config(text= "Chữ nhận dạng",width=22)
        self.label_iamge.config(image="",text="Hình ảnh")
        



    def processing(self):
        model = E2E()
        image_r = cv2.imread(str(self.filename))
        if(self.label_time.get() == "CNN"):
            image = model.predict(image_r,"CNN")
            cv2.imshow('License Plate', image)
        else: 
            image = model.predict(image_r,"SVM")
    
            # đoạn này gắn hình biển số xe dô thôi
        self.imageLpRegion = Image.open("step1.png")
        self.imageLpRegion_1 = self.imageLpRegion.resize((170,40),Image.ANTIALIAS)
        self.imageLpRegion_ex = ImageTk.PhotoImage(self.imageLpRegion_1)
        self.label_plate.config(image=self.imageLpRegion_ex,width=170)

            # đoạn này gắn biển số xe sau khi đã đổi màu
        self.imageLpRegion_color = Image.open("step2_2.png")
        self.imageLpRegion_1_color = self.imageLpRegion_color.resize((170,40),Image.ANTIALIAS)
        self.imageLpRegion_ex_color = ImageTk.PhotoImage(self.imageLpRegion_1_color)
        self.label_plate_color.config(image=self.imageLpRegion_ex_color,width=170)

        if(self.label_time.get() == "CNN"):
            self.license_plate = model.get_license_plate("CNN")
            print(self.license_plate)
            self.edit_character.config(text= self.license_plate)
        else: 
            self.license_plate = model.get_license_plate("SVM")
            print(self.license_plate)
            self.edit_character.config(text= self.license_plate)

            # gắn chữ số đã nhận dạng
        

        
    def open(self):
        # global resize_image, filename
        self.filename = filedialog.askopenfilename(initialdir="/gui/images",title="tải ảnh cần nhận diện",filetypes=(("jpg files","*.jpg"),("png files","*.png")))
        self.image_main = Image.open(self.filename)
        self.image_main_1 = self.image_main.resize((500,500),Image.ANTIALIAS)
        self.resize_image = ImageTk.PhotoImage(self.image_main_1)
        self.label_iamge.config(image=self.resize_image)


    def insert(self,image):
        self.label_plate.config(image=image)

  

if __name__ == '__main__':
    main = Main()
    main.mainloop()
