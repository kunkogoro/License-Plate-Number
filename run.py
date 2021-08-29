
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import font
from tkinter.font import families
from PIL import ImageTk,Image
import os
import glob
import tkinter.messagebox as mbox

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
        self.geometry("700x600+%d+%d"%(srcW/2-350,srcH/2-350))
        self.resizable(width=False,height=False)
        self.filenameOld = ""


        self.frame_result = LabelFrame(self,text="Kết quả",padx=10,pady=10,width=200,height=500)
        # frame_result.pack(side=LEFT,fill="both",expand="yes")
        self.frame_result.grid(row=0,column=0,sticky=N)
        # button nhấn vào để load hình ảnh

        self.frame_plate = LabelFrame(self.frame_result,text="Biển số",padx=3,pady=3,width=400,font="arial 10 bold",height=50)
        self.frame_plate.grid(row=1,column=0)
        self.label_plate = Label(self.frame_plate,text="Biển số",width=22)
        self.label_plate.pack()


        self.frame_plate_color = LabelFrame(self.frame_result,text="Đổi màu",padx=3,pady=3,width=200,height=100,font="arial 10 bold")
        self.frame_plate_color.grid(row=2,column=0)
        self.label_plate_color = Label(self.frame_plate_color,text="Đổi màu",width=22)
        self.label_plate_color.pack()

        self.frame_plate_contour = LabelFrame(self.frame_result,text="Xác định chữ",padx=3,pady=3,width=200,height=100,font="arial 10 bold")
        self.frame_plate_contour.grid(row=3,column=0)
        self.label_plate_contour = Label(self.frame_plate_contour,text="Xác định chữ",width=22)
        self.label_plate_contour.pack()


        self.frame_character = LabelFrame(self.frame_result,text="Kết quả nhận dạng",padx=3,pady=3,width=200,height=100,font="arial 10 bold")
        self.frame_character.grid(row=4,column=0)
        self.edit_character = Label(self.frame_character,width=22,text="Chữ nhận dạng")
        self.edit_character.pack()

        self.frame_time = LabelFrame(self.frame_result,text="Thời gian",font="arial 10 bold",padx=3,pady=3,width=200,height=100)
        self.frame_time.grid(row=5,column=0)
        self.edit_time = Label(self.frame_time,width=22,text="Thời gian")
        self.edit_time.pack()

        self.frame_model = LabelFrame(self.frame_result,text="Model",font="arial 10 bold",padx=10,pady=10,width=200,height=100)
        self.frame_model.grid(row=6,column=0)
        self.label_time = ttk.Combobox(self.frame_model,width=22)
        self.label_time["value"] = ("CNN","SVM")
        self.label_time.current(0)
        self.label_time.pack()


        self.btn = Button(self.frame_result,text="Lấy hình",padx=50,pady=7,font="arial 10 bold",command=self.open)
        self.btn.grid(row=7,column=0,pady=7)

        self.btn_processing = Button(self.frame_result,text="Thực hiện",padx=50,pady=7,font="arial 10 bold",command=self.processing)
        self.btn_processing.grid(row=8,column=0,pady=7)

        self.btn_reset = Button(self.frame_result,text="Làm mới",padx=50,pady=7,font="arial 10 bold",command=self.reset)
        self.btn_reset.grid(row=9,column=0,pady=7)

        self.btn_exit = Button(self.frame_result,text="Thoát",padx=50,pady=7,font="arial 10 bold",command=self.close)
        self.btn_exit.grid(row=10,column=0,pady=7)



        self.frame_image = LabelFrame(self,text="Hình ảnh",padx=5,pady=5,width=500,height=500)
        # frame_image.pack(side=RIGHT)
        self.frame_image.grid(row=0,column=1)
        self.label_iamge = Label(self.frame_image,text="Hình ảnh",width=500,height=500)
        self.label_iamge.pack()
        

    def close(self):
        main.destroy()


    def resetAll(self):
        self.label_plate.config(image="",text="Biển số",width=22)
        self.label_plate_color.config(image="",text="Đổi màu",width=22)
        self.edit_character.config(text= "Chữ nhận dạng",width=22)
        self.label_iamge.config(image="",text="Hình ảnh")
        self.edit_time.config(image="",text="Thời gian")
        self.label_plate_contour.config(image="",text="Xác định chữ",width=22)
    def reset(self):
        self.label_plate_color.config(image="",text="Đổi màu",width=22)
        self.edit_character.config(text= "Chữ nhận dạng",width=22)
        self.label_plate.config(image="",text="Biển số",width=22)
        self.edit_time.config(image="",text="Thời gian")
        self.label_plate_contour.config(image="",text="Xác định chữ",width=22)
        
        
    def processing(self):
        if(self.filename == "" and self.filenameOld == ""):
            mbox.showerror("Lỗi hình ảnh", "Vui lòng chọn hình ảnh")
        if(self.filenameOld != ""):

            self.reset()

            self.start = time.time()

            model = E2E()
            image_r = cv2.imread(str(self.filenameOld))
            if(self.label_time.get() == "CNN"):
                self.image = model.predict(image_r,"CNN")
                self.end = time.time()
            else: 
                self.image = model.predict(image_r,"SVM")
                self.end = time.time()

            # gắn hình kết quả
            self.image_result = Image.open("result.png")
            self.resize_image = self.image_result.resize((500,500),Image.ANTIALIAS)
            self.result = ImageTk.PhotoImage(self.resize_image)
            self.label_iamge.config(image=self.result)
        
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


            # đoạn này tạo các viền bao quanh các chữ
            self.imageLpRegion_contour = Image.open("step3.png")
            self.imageLpRegion_2_color = self.imageLpRegion_contour.resize((170,40),Image.ANTIALIAS)
            self.imageLpRegion_ex_contour = ImageTk.PhotoImage(self.imageLpRegion_2_color)
            self.label_plate_contour.config(image=self.imageLpRegion_ex_contour,width=170)

           
            self.license_plate = model.get_license_plate()
            print(self.license_plate)
            self.edit_character.config(text= self.license_plate)

            self.license_plate = model.get_license_plate()
            self.time = '%.2f s' % (self.end - self.start)
            print('Thực hiện %.2f s' % (self.end - self.start))
            self.edit_time.config(text= self.time)


           

            # gắn chữ số đã nhận dạng

            
        

        
    def open(self):
        # global resize_image, filename
        self.filename = filedialog.askopenfilename(initialdir="/Desktop",title="tải ảnh cần nhận diện",filetypes=(("jpg files","*.jpg"),("png files","*.png")))
        self.image_main = Image.open(self.filename)
        self.image_main_1 = self.image_main.resize((500,500),Image.ANTIALIAS)
        self.resize_image = ImageTk.PhotoImage(self.image_main_1)
        self.label_iamge.config(image=self.resize_image)
        if(self.filename != ""):
            self.filenameOld = self.filename
            self.reset()


    def insert(self,image):
        self.label_plate.config(image=image)

  

if __name__ == '__main__':
    main = Main()
    main.mainloop()
