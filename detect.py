import data_utils as utils
import cv2
import numpy as np


class detectNumberPlate(object):
    def __init__(self, threshold=0.5):
        args = utils.get_arguments()
        self.weight_path = args.weight_path
        self.cfg_path = args.config_path
        self.labels = utils.get_labels(args.classes_path)

        # độ tin cậy
        self.threshold = threshold

        # Load model
        # hàm này mục đích là tải các mô hình đào tạo từ trước
        self.model = cv2.dnn.readNet(model=self.weight_path, config=self.cfg_path)

    # mục đích của hàm này là tìm ra tọa độ và các kích thước của đối tượng
    def detect(self, image):
        # mảng dùng để chứa các thông tin tọa độ và chiều dài, chiều cao
        boxes = []
        # classes_id: chỉ số lớp có độ tin cậy cao nhất
        classes_id = []
        # mảng chức độ tin cậy tương ứng khi phát hiện 1 đối tượng, nó tương ứng với boxes[]
        confidences = []
        scale = 0.00392


        # hàm này mục đích là xử lý dữ liệu hình ảnh phù hợp rồi đưa vào mô hình
        # image: hình ảnh, scalefactor:chia tỷ lệ hình ảnh của mình theo một số yếu tố,size: kích thước không gian mà Mạng nơ-ron phù hợp mong đợi
        # kết quả: hình ảnh đầu vào sau khi trừ trung bình, chuẩn hóa và hoán đổi kênh.
        #scalefactor: Giá trị này chia tỷ lệ hình ảnh theo giá trị được cung cấp. Nó có giá trị mặc định là 1 có nghĩa là không có quy mô nào được thực hiện.
        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True, crop=False)
        height, width = image.shape[:2]

        # take image to model
        # cho input vào cãi đã
        self.model.setInput(blob)

        # run forward
        # một danh sách lồng nhau chứa thông tin về tất cả các đối tượng 
        # được phát hiện bao gồm tọa độ x và y của tâm đối tượng được phát hiện,
        #  chiều cao và chiều rộng của hộp giới hạn, độ tin cậy và điểm cho tất cả các lớp đối tượng
        # outputs có dạng là ( 1, 1000, 1, 1)
        outputs = self.model.forward(utils.get_output_layers(self.model))

        #range(): tạo ra một mảng các số từ 0 đến số đó 
        for output in outputs:
            for i in range(len(output)):
                # scores: độ tin cậy
                scores = output[i][5:]

                #class_id tọa độ điểm lớn nhất
                class_id = np.argmax(scores)
                # độ tin cậy 
                confidence = float(scores[class_id])

                # những đối tượng nào có độ tin cậy lớn hơn chỉ định thì lấy
                if confidence > self.threshold:
                    # coordinate of bounding boxes
                    # tọa độ tâm của đối tượng được phát hiện
                    center_x = int(output[i][0] * width)
                    center_y = int(output[i][1] * height)

                    # chiều rộng, chiều cao được phát hiện
                    detected_width = int(output[i][2] * width)
                    detected_height = int(output[i][3] * height)

                    # tọa đọ góc trên cùng bến trái
                    x_min = center_x - detected_width / 2
                    y_min = center_y - detected_height / 2

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    classes_id.append(class_id)
                    confidences.append(confidence)
        ## khác phục nhìu đối tượng chống chéo lên nhau
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.threshold, nms_threshold=0.4)


        coordinates = []
        for i in indices:
            index = i[0]
            x_min, y_min, width, height = boxes[index]
            x_min = round(x_min)
            y_min = round(y_min)

            coordinates.append((x_min, y_min, width, height))

        return coordinates
