import Models as Nets
import Trainer as Trainer


if __name__ == "__main__":
    path = r"D:\数据集\CelebaA\100k_version\12"
    net = Nets.PNet
    train = Trainer.Trainer(net, path, 512)
    train.train("pnet", 0.01)
