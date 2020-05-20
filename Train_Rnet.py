import MTCNN_Version_2.Nets as Nets
import MTCNN_Version_2.Trainer as Trainer


if __name__ == "__main__":
    net = Nets.RNet
    train = Trainer.Trainer(net, r"D:\Aaron\liewei tech\Datasets\Dataset_MTCNN_Study\dataset_landmarkers\24")
    train.train("rnet", 0.001)
