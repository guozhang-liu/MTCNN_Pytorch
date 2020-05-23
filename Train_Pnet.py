import MTCNN_Version_2.Nets as Nets
import MTCNN_Version_2.Trainer as Trainer


if __name__ == "__main__":
    path = r"./"
    net = Nets.PNet
    train = Trainer.Trainer(net, path)
    train.train("pnet", 0.01)
