import torch
from Sampling import DatasetFaces
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
import MTCNN_Version_2.Nets as Nets
import matplotlib.pyplot as plt
import os
from tqdm.autonotebook import tqdm


class Trainer:
    def __init__(self, net, path, isLandmarks=False):
        self.landmarks = isLandmarks
        self.batch_size = 512
        self.net = net().cuda() if torch.cuda.is_available() else net().cpu()
        self.dataset = DatasetFaces(path, self.landmarks)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.optimizer = Adam(self.net.parameters())
        self.loss_confidence = nn.BCELoss()
        self.loss_offset = nn.MSELoss()
        self.model_path = "./Weights"

    def train(self, net_name, stop_value):
        global loss_landmarks
        if os.path.exists("./Weights/{}.pth".format(net_name)):
            self.net.load_state_dict(torch.load("./Weights/{}.pth".format(net_name), map_location="cpu"))
            print("Net_Exits")
        loss = 0
        losses = []
        while True:
            dataloader = tqdm(self.dataloader)
            for i, data in enumerate(dataloader):
                img_data, confidence, offsets = data[0].cuda(), data[1].cuda(), data[2].cuda()
                output_data = self.net(img_data)
                out_confidence, out_offsets = output_data[0], output_data[1]
                out_confidence = out_confidence.view(-1, 1)  # 将输出置信度展平为N行1列，以便后续筛选
                out_offsets = out_offsets.view(-1, 4)
                # lt=less than 用置信度为0，1的来训练分类
                mask_category = torch.lt(confidence, 2)
                confidence_category, out_confidence_category = confidence[mask_category], out_confidence[mask_category]
                loss_category = self.loss_confidence(out_confidence_category, confidence_category)

                # gt=greater than 用置信度1，2来训练坐标偏移率
                mask_coordinate = torch.gt(confidence, 0)
                offsets_coordinate = torch.masked_select(offsets, mask_coordinate)
                out_offsets_coordinate = torch.masked_select(out_offsets, mask_coordinate)
                loss_coordinate = self.loss_offset(out_offsets_coordinate, offsets_coordinate)

                # gt=greater than 用置信度1，2来训练landmarks
                if self.landmarks:
                    output_landmarks = output_data[2]
                    output_landmarks = output_landmarks.view(-1, 10)
                    landmarks_data = data[3].cuda()
                    mask_landmarks = torch.gt(confidence, 0)
                    landmarks = torch.masked_select(landmarks_data, mask_landmarks)
                    out_landmarks_points = torch.masked_select(output_landmarks, mask_landmarks)
                    loss_landmarks = self.loss_offset(landmarks, out_landmarks_points)
                    loss = loss_coordinate + loss_category + loss_landmarks

                else:
                    loss = loss_coordinate + loss_category

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss)
                if self.landmarks:
                    dataloader.set_description("loss_category:{:.4f}---loss_coordinate:{:.4f}---loss_landmarks:{:.4f}"
                                               "---loss:{:.4f}".format(loss_category, loss_coordinate, loss_landmarks, loss))
                else:
                    dataloader.set_description("loss_category:{:.4f}---loss_coordinate:{:.4f}------loss:{:.4f}".
                                                format(loss_category, loss_coordinate, loss))

                del img_data, confidence, confidence_category, offsets, offsets_coordinate, out_offsets_coordinate, \
                    out_offsets, out_confidence_category, out_confidence, i, loss_category, loss_coordinate, output_data

            torch.save(self.net.state_dict(), "./Weights/{}.pth".format(net_name))
            print("Saved Successfully")
            if loss < stop_value:
                break


if __name__ == "__main__":
    train = Trainer(Nets.ONet, r"D:\Aaron\liewei tech\Datasets\Dataset_MTCNN_Study\dataset_landmarkers\48", isLandmarks=True)
    train.train("onet", 0.1)
