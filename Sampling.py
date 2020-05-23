from torch.utils.data import Dataset
import numpy as np
import os
import torch
import PIL.Image as Image


class DatasetFaces(Dataset):
    def __init__(self, path, isLandmarks=False):
        self.path = path
        self.dataset = []
        for file in ["positive_target.txt", "negative_target.txt", "part_target.txt"]:
            self.dataset.extend(open(os.path.join(path, "{}".format(file))).readlines())
        self.landmarks = isLandmarks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        target_split = self.dataset[index].strip().split(" ")
        confidence = torch.tensor([float(target_split[1])], dtype=torch.float32)  # 置信度
        offsets = torch.tensor([float(target_split[2]), float(target_split[3]), float(target_split[4]),
                                float(target_split[5])], dtype=torch.float32)  # 偏移率
        img_path = os.path.join(self.path, str(target_split[0]))
        img = torch.tensor(np.array(Image.open(img_path)), dtype=torch.float32).permute(2, 0, 1)
        img_data = (img / 255 - 0.5) / 0.5
        if self.landmarks:
            landmarks = torch.tensor([float(target_split[6]), float(target_split[7]), float(target_split[8]),
                                     float(target_split[9]), float(target_split[10]), float(target_split[11]),
                                     float(target_split[12]), float(target_split[13]), float(target_split[14]),
                                     float(target_split[15])], dtype=torch.float32)

            return img_data, confidence, offsets, landmarks

        return img_data, confidence, offsets


if __name__ == "__main__":
    dataset = DatasetFaces(r"D:\Aaron\liewei tech\Datasets\Dataset_MTCNN_Study\dataset_landmarkers\48", isLandmarks=True)
    print(dataset.__getitem__(1000))
    # dataloader = DataLoader(dataset, batch_size=10001, shuffle=False)
    # for i in dataloader:
    #     mean = torch.mean(i, dim=(0, 2, 3))
    #     std = torch.std(i, dim=(0, 2, 3))
    #     print(mean, std)

