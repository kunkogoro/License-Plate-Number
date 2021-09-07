import keras
import numpy as np

import config
from data_provider import Datasets
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


class CNN_Model(object):
    def __init__(self, trainable=True):
        self.batch_size = config.BATCH_SIZE
        self.trainable = trainable
        self.num_epochs = config.EPOCHS
       

        # Building model
        # xây dựng model
        self._build_model()

        # Input data
        if trainable:
            self.model.summary()
            self.data = Datasets()

        #Ở hàm này  sử dụng để training models như thuật toán train qua optimizer như Adam, SGD, RMSprop,..
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['accuracy'])
        # self.train()
        # self.evaluate_model()


    def evaluate_model(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    def _build_model(self):
        # CNN model
        # thư viện Keras, hỗ trợ xây dựng CNN
        # Khởi tạo models Sequential ( )
        self.model = Sequential()
        # Tạo Convolutionnal Layers : Conv2D là convolution dùng để lấy feature từ ảnh với các tham số :
        # 32: filters là số class dùng để phân loại là các số từ 0-9 và chữ A-Z và background(nhiễu)
        # kernel_size : kích thước window search trên ảnh
        # strides : số bước nhảy trên ảnh
        #padding : có thể là "valid" hoặc "same". Với same thì có nghĩa là padding =1.
        #activation : chọn activation như linear, softmax, relu, tanh, sigmoid.
        # hàm rule công thức: max(0,x)
        # ưu điểm hội tụ nhanh hơn sơ với sigmoid và tanh
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))

        # Pooling Layers: sử dụng để làm giảm param khi train, nhưng vẫn giữ được đặc trưng của ảnh.
        #pool_size : kích thước ma trận để lấy max hay average (MaxPooling2D lấy max)
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        ## Flatten layer chuyển từ tensor sang vector
        self.model.add(Flatten())
       # Dense ( ): Layer này cũng như một layer neural network bình thường
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        #units : số chiều output, như số class sau khi train (0,1,2,...).
        self.model.add(Dense(32, activation='softmax'))

    def train(self):
        # reduce learning rate
        #giảm learning mỗi khi metrics không cải thiện
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1, )
        # Model Checkpoint
        #lưu lại model sau mỗi epoch
        cpt_save = ModelCheckpoint('./weights/weight2.h5', save_best_only=True, monitor='val_accuracy', mode='max')

        print("Training......")
        trainX, trainY = self.data.gen()
        trainX = np.array(trainX)

        #fit()Đào tạo mô hình cho một số kỷ nguyên cố định (các lần lặp lại trên một tập dữ liệu).
        # Callbacks: Khi models chúng ta lớn khi training thì gặp lỗi ta muốn lưu lại models để chạy lại thì ta sử dụng callbacks
        # tranX: dữ liệu đầu vào
        # tranY: dữ liệu mục tiêu
        #epochs:  số lần duyệt qua hết số lượng mẫu trong tập huấn luyện.
        # verbose: Chế độ chi tiết
        #validation_split: chia dữ liệu ra 2 phần: train và test 0.15 là tran 85% test 15%
        #batch_size: thể hiện số lượng mẫu sử dụng cho mỗi lần cập nhật trọng số .
        self.history =  self.model.fit(trainX, trainY, validation_split=0.15, callbacks=[cpt_save, reduce_lr], verbose=1,
                       epochs=self.num_epochs, shuffle=True, batch_size=self.batch_size)

if __name__ == '__main__':
    main = CNN_Model(trainable=True)
    


