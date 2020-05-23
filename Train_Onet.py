import Models as Nets
import Trainer as Trainer


if __name__ == "__main__":
    path = r"D:\数据集\CelebaA\100k_version\48"
    net = Nets.ONet
    train = Trainer.Trainer(net, path, 512, isLandmarks=True)
    train.train("onet", 0.0005)
