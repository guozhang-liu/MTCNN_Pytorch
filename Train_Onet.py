import MTCNN_Version_2.Nets as Nets
import MTCNN_Version_2.Trainer as Trainer


if __name__ == "__main__":
    net = Nets.ONet
    train = Trainer.Trainer(net, r"D:\Aaron\liewei tech\Datasets\Dataset_MTCNN_Study\dataset_landmarkers\48",
                            isLandmarks=True)

    train.train("onet", 0.0009)
