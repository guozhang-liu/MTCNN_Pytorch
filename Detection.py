import torch
import torchvision.transforms as trans
import time
import PIL.Image as Image
import numpy as np
from Tools.utilis import NMS, Convert_to_square
import PIL.ImageDraw as ImageDraw
import cv2
import Models

"""
1.pnet的侦测通过切片形式完成，提高了p网络侦测效率
2.图片输出通过opencv
"""

class Detector:
    PNet_path = r"./Weights/pnet.pth"
    RNet_path = r"./Weights/rnet.pth"
    ONet_path = r"./Weights/onet.pth"

    def __init__(self, P_path=PNet_path, R_path=RNet_path, O_path=ONet_path, isCUDA=True):
        self.pnet = Models.PNet()
        self.rnet = Models.RNet()
        self.onet = Models.ONet()
        self.cuda = isCUDA

        if self.cuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        #  测试在GPU上进行速度较慢，即加载到CPU上
        self.pnet.load_state_dict(torch.load(P_path, map_location="cpu"))
        self.rnet.load_state_dict(torch.load(R_path, map_location="cpu"))
        self.onet.load_state_dict(torch.load(O_path, map_location="cpu"))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.img_transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def detect(self, img):
        start_time = time.time()
        pnet_boxes = self.pnet_detection(img)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        pnet_time = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.rnet_detection(img, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detection(img, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        onet_time = end_time - start_time
        time_total = pnet_time + rnet_time + onet_time

        print("Pnet:{:.3f}s, Rnet:{:.3f}s, Onet:{:.3f}s, Total:{:.3f}s".
              format(pnet_time, rnet_time, onet_time, time_total))

        return onet_boxes

    def pnet_detection(self, img):
        scale = 0.7  # 第一张图尺寸为原图尺寸
        Boxes = []
        w, h = img.size
        img = img.resize((int(scale*w), int(scale*h)))  # 抛弃图像金字塔第一张
        min_side_len = min(w, h)
        while min_side_len > 12:  # 当最小边长大于12才继续循环
            if self.cuda:
                img_data = self.img_transform(img).cuda()
            else:
                img_data = self.img_transform(img).cpu()
            img_data = img_data.unsqueeze(0)
            output = self.pnet(img_data)
            confidence, offsets = output[0][0, 0].data, output[1][0].data
            confidence_mask = torch.gt(confidence, 0.6)  # 置信度大于0.6的掩膜，置信度大于0.6的为1
            indexes = torch.nonzero(confidence_mask)  # 获得1（置信度大于0.6）部分的坐标

            offsets_index = offsets[:, indexes[:, 0], indexes[:, 1]].permute(1, 0)  # 218*4

            confidence_index = confidence[indexes[:, 0], indexes[:, 1]].reshape(-1, 1)  # 218*1

            boxes = self.draw_boxes(indexes, scale, offsets_index, confidence_index)  # 218*5

            Boxes.append(boxes)

            scale *= 0.7
            _w, _h = int(w*scale), int(h*scale)  # 构建图像金字塔
            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)
        Boxes = torch.cat(Boxes, 0).cpu()  # 将每次图像金字塔所得数组展开拼接成-1*5

        return NMS(np.array(Boxes), 0.3)

    def draw_boxes(self, start_index, scale, offsets, confidence, stride=2, side_len=12):
        # 反算出建议框
        _x1 = ((start_index[:, 1]*stride)/scale).reshape(-1, 1).int()
        _y1 = ((start_index[:, 0]*stride)/scale).reshape(-1, 1).int()
        _x2 = ((start_index[:, 1]*stride+side_len)/scale).reshape(-1, 1).int()
        _y2 = ((start_index[:, 0]*stride+side_len)/scale).reshape(-1, 1).int()

        ow = _x2 - _x1  # 宽
        oh = _y2 - _y1  # 高

        # 反算出原框
        x1 = (_x1 + offsets[:, 0].reshape(-1, 1)*ow)
        y1 = (_y1 + offsets[:, 1].reshape(-1, 1)*oh)
        x2 = (_x2 + offsets[:, 2].reshape(-1, 1)*ow)
        y2 = (_y2 + offsets[:, 3].reshape(-1, 1)*oh)

        return torch.cat([x1, y1, x2, y2, confidence], 1)

    def rnet_detection(self, image, pnet_boxes):
        _pic_box = []  # rnet输入图片数据集
        _pnet_boxes = Convert_to_square(pnet_boxes)  # 转正方形
        for _box in _pnet_boxes:  # 获得rnet输出转为正方形的框坐标
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 将原图上裁剪下pnet输出框
            img = img.resize((24, 24))  # 缩放为rnet输入图片尺寸24*24
            img_tensor = self.img_transform(img)
            _pic_box.append(img_tensor)

        rnet_inputdata = torch.stack(_pic_box)  # 将_pic_box的列表转为输入rnet的四维tensor

        # 数据输入网络
        if self.cuda:
            rnet_inputdata = rnet_inputdata.cuda()
        confidence, offsets = self.rnet(rnet_inputdata)
        confidence, offsets = confidence.cpu().data.numpy(), offsets.cpu().data.numpy()

        indexes, _ = np.where(confidence > 0.6)

        """
        切片法
        """
        # _box = _pnet_boxes[indexes]
        # _x1 = (_box[:, 0]).reshape(-1, 1)
        # _y1 = (_box[:, 1]).reshape(-1, 1)
        # _x2 = (_box[:, 2]).reshape(-1, 1)
        # _y2 = (_box[:, 3]).reshape(-1, 1)
        #
        # ow = _x2 - _x1
        # oh = _y2 - _y1
        #
        # x1 = (_x1 + ow * offsets[indexes][:, 0].reshape(-1, 1))
        # y1 = (_y1 + oh * offsets[indexes][:, 1].reshape(-1, 1))
        # x2 = (_x2 + ow * offsets[indexes][:, 2].reshape(-1, 1))
        # y2 = (_y2 + oh * offsets[indexes][:, 3].reshape(-1, 1))
        # cls = confidence[indexes]
        #
        # boxes = torch.cat([torch.tensor(x1, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32),
        #                    torch.tensor(x2, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32),
        #                    torch.tensor(cls, dtype=torch.float32)], 1)

        boxes = []
        for index in indexes:
            _box = _pnet_boxes[index]
            _x1 = float(_box[0])
            _y1 = float(_box[1])
            _x2 = float(_box[2])
            _y2 = float(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = float(_x1 + ow * offsets[index][0])
            y1 = float(_y1 + oh * offsets[index][1])
            x2 = float(_x2 + ow * offsets[index][2])
            y2 = float(_y2 + oh * offsets[index][3])
            cls = confidence[index][0]
            boxes.append([x1, y1, x2, y2, cls])

        return NMS(np.array(boxes), 0.3)

    def onet_detection(self, image, rnet_boxes):
        _pic_box = []  # onet输入图片数据集
        _rnet_boxes = Convert_to_square(rnet_boxes)  # 转正方形
        for _box in _rnet_boxes:  # 获得onet输出转为正方形的框坐标
            _x1 = float(_box[0])
            _y1 = float(_box[1])
            _x2 = float(_box[2])
            _y2 = float(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 将原图上裁剪下rnet输出框
            img = img.resize((48, 48))  # 缩放为onet输入图片尺寸48*48

            img_tensor = self.img_transform(img)
            _pic_box.append(img_tensor)
        onet_inputdata = torch.stack(_pic_box)  # 将_pic_box的列表转为输入onet的四维tensor

        # 数据输入网络
        if self.cuda:
            onet_inputdata = onet_inputdata.cuda()
        confidence, offsets, landmarks_offsets = self.onet(onet_inputdata)
        confidence, offsets, landmarks_offsets = confidence.cpu().data.numpy(), offsets.cpu().data.numpy(), \
                                                 landmarks_offsets.cpu().data.numpy()

        boxes = []
        indexes, _ = np.where(confidence > 0.9)

        for index in indexes:
            _box = _rnet_boxes[index]
            _x1 = float(_box[0])
            _y1 = float(_box[1])
            _x2 = float(_box[2])
            _y2 = float(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = float(_x1 + ow * offsets[index][0])
            y1 = float(_y1 + oh * offsets[index][1])
            x2 = float(_x2 + ow * offsets[index][2])
            y2 = float(_y2 + oh * offsets[index][3])

            # 反算出landmarks
            px1 = float(_x1 + ow * landmarks_offsets[index][0])
            py1 = float(_y1 + oh * landmarks_offsets[index][1])
            px2 = float(_x1 + ow * landmarks_offsets[index][2])
            py2 = float(_y1 + oh * landmarks_offsets[index][3])
            px3 = float(_x1 + ow * landmarks_offsets[index][4])
            py3 = float(_y1 + oh * landmarks_offsets[index][5])
            px4 = float(_x1 + ow * landmarks_offsets[index][6])
            py4 = float(_y1 + oh * landmarks_offsets[index][7])
            px5 = float(_x1 + ow * landmarks_offsets[index][8])
            py5 = float(_y1 + oh * landmarks_offsets[index][9])

            cls = confidence[index][0]

            boxes.append([x1, y1, x2, y2, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5, cls])

        return NMS(np.array(boxes), 0.3, isMin=True)


if __name__ == "__main__":
    x = time.time()
    with torch.no_grad() as grad:
        image_file = r"./Test_Pics/1.jpg"
        detector = Detector()
        image = cv2.imread(image_file)
        img = image[..., ::-1]
        img = Image.fromarray(img)
        boxes = detector.detect(img)

        for box in boxes:
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

            px1 = float(box[4])
            py1 = float(box[5])
            px2 = float(box[6])
            py2 = float(box[7])
            px3 = float(box[8])
            py3 = float(box[9])
            px4 = float(box[10])
            py4 = float(box[11])
            px5 = float(box[12])
            py5 = float(box[13])

            for point in [(px1, py1), (px2, py2), (px3, py3), (px4, py4), (px5, py5)]:
                cv2.line(image, (point[0]-4, point[1]), (point[0]+4, point[1]), color=(0, 255, 255), thickness=2)
                cv2.line(image, (point[0], point[1]-4), (point[0], point[1]+4), color=(0, 255, 255), thickness=2)

        y = time.time()
        print(y - x)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", image)
        cv2.waitKey(0)


