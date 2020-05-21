import cv2
import PIL.Image as Image
from Detection import Detector


video = cv2.VideoCapture('./Test_Pics/1.mp4')
size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # opencv为N,W,H,C
save_path = "./Output/video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 编码格式
fps = 30  # 帧率
out = cv2.VideoWriter(save_path, fourcc, fps, size)

c = 0
while True:
    ret, frame = video.read()
    if ret:
        timeF = 1  # detecting for one frame
        if c % timeF == 0:
            img = frame[..., ::-1]
            img = Image.fromarray(img)
            boxes = Detector().detect(img.copy())
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 坐标必须是整型
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
            out.write(frame)
        c += 1
        cv2.imshow("camera", frame)
        if cv2.waitKey(30) & 0xFF == ord("a"):
            break

video.release()
cv2.destroyAllWindows()

