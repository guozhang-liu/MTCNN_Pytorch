import MTCNN_Version_2.Nets as Nets
import MTCNN_Version_2.Trainer as Trainer


if __name__ == "__main__":
    path = r"./"
    net = Nets.ONet
    train = Trainer.Trainer(net, path, isLandmarks=True)
    train.train("onet", 0.0005)
