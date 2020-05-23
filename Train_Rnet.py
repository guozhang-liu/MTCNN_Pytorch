import Nets as Nets
import Trainer as Trainer


if __name__ == "__main__":
    path = r"D:\数据集\CelebaA\100k_version\24"
    net = Nets.RNet
    train = Trainer.Trainer(net, path)
    train.train("rnet", 0.001)
