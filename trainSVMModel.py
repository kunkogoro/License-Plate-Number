import cv2
import os
import numpy as np
import glob
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score
import matplotlib.pyplot as plt
import itertools
from joblib import dump, load
import pickle

digit_w = 30
digit_h = 60

write_path="./weights/modelSVM1.joblib"


def get_digit_data(path,number = 0):#:, digit_list, label_list):

    digit_list = []
    label_list = []
  
    alpha = ["A","B","C","D","E","F","G","H","K","L","M","N","P","R","S","T","U","V","X","Y","Z"]

    for number in range(10):
        # print(number)
        i=0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
           
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)

            digit_list.append(img)
            label_list.append([int(number)])

    for number in alpha:
        #number = chr(number)
        # print(number)
        i=0
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            img = img.reshape(-1, digit_h * digit_w)

            # print(img.shape)

            digit_list.append(img)
            label_list.append([number])
           
    return  digit_list, label_list

#lấy dữ liệu train
digit_path = "D:/OneDrive/Desktop/dataTrainSVM/"
digit_list_train, label_list_train = get_digit_data(digit_path)

digit_list_train = np.array(digit_list_train, dtype=np.float32)
digit_list_train = digit_list_train.reshape(-1, digit_h * digit_w)

label_list_train = np.array(label_list_train)
label_list_train = label_list_train.reshape(-1, 1)

clf = svm.SVC()
clf.fit(digit_list_train, label_list_train)

# luuw lai
dump(clf, write_path)


# #lấy dữ liệu test
digit_path_test = "D:/OneDrive/Desktop/dataTestSVM/"
digit_list_test, label_list_test = get_digit_data(digit_path_test)

digit_list_test = np.array(digit_list_test, dtype=np.float32)
digit_list_test = digit_list_test.reshape(-1, digit_h * digit_w)

label_list_test = np.array(label_list_test)
label_list_test = label_list_test.reshape(-1, 1)



# đánh giá model
ypred = clf.predict(digit_list_test)
acc = accuracy_score(label_list_test,ypred)
recall_score = recall_score(label_list_test, ypred, average='macro')
precision_score = precision_score(label_list_test, ypred, average='macro')
print("Score: ", acc)
print('Recall:', recall_score)
print('Precision:', precision_score)
cm = confusion_matrix(label_list_test, ypred)
print("Confusion Matrix:",cm)


classes=["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","K","L","M","N","P","R","S","T","U","V","X","Y","Z"]


normalize = True
cmap=plt.cm.Blues
if normalize:
  cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

fig = plt.figure(figsize=(20,16))
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Normalized confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()