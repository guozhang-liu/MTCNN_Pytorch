import MTCNN_Version_2.Nets as Nets
import MTCNN_Version_2.Trainer as Trainer


if __name__ == "__main__":
    path = r"./"
    net = Nets.RNet
    train = Trainer.Trainer(net, path)
    train.train("rnet", 0.001)
