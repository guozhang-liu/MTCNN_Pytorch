import Tools.utilis as utilis
import os
import traceback
import PIL.Image as Image
import numpy as np

pics_path = r"D:\数据集\CelebaA\img\img_celeba\img_celeba"
targets_path = r"D:\数据集\CelebaA\Anno\list_bbox_celeba.txt"
landmarks_path = r"D:\数据集\CelebaA\Anno\list_landmarks_celeba.txt"
save_path = r"D:\数据集\CelebaA\100k_version"
float_num = [0.1, 0.5, 0.5, 0.5, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99]


def generator(pics_size, stop_value, isLandmarks=False):
    positive_samples_path = os.path.join(save_path, str(pics_size), "positive_samples")  # 三种样本所对应的路径
    part_samples_path = os.path.join(save_path, str(pics_size), "part_samples")
    negative_samples_path = os.path.join(save_path, str(pics_size), "negative_samples")

    with open(landmarks_path) as f:
        landmark = f.readlines()

        for dirs in [positive_samples_path, part_samples_path, negative_samples_path]:  # 创建12、24、48所对应的正、部分、负样本文件夹
            if not os.path.exists(dirs):
                os.makedirs(dirs)

        try:
            positive_target_path = os.path.join(save_path, str(pics_size), "positive_target.txt")  # 创建三种样本标签的txt
            part_target_path = os.path.join(save_path, str(pics_size), "part_target.txt")
            negative_target_path = os.path.join(save_path, str(pics_size), "negative_target.txt")

            positive_target = open(positive_target_path, "w")
            part_target = open(part_target_path, "w")
            negative_target = open(negative_target_path, "w")

            positive_count = 0
            part_count = 0
            negative_count = 0

            for i, line in enumerate(open(targets_path)):
                if i < 2:
                    continue
                x1, y1, x2, y2 = float(line.split()[1].strip()), float(line.split()[2].strip()), float(
                    line.split()[1].strip()) + \
                                 float(line.split()[3].strip()), float(line.split()[2].strip()) + 0.9 * float(
                    line.split()[4].strip())  # 原图框四个坐标

                w, h = float(x2 - x1), float(y2 - y1)  # 原图框长宽
                center_axes = [float(x1 + w / 2), float(y1 + h / 2)]  # 原图中心点坐标

                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                origin_box = [[x1, y1, x2, y2]]  # 原框坐标点，计算IOU使用。Tools中的IOU中boxes输入为二维数组
                img_path = os.path.join(pics_path, line.split()[0])  # origin中图片打开路径
                img = Image.open(img_path)  # type:Image.Image
                img_w, img_h = img.size  # 得到图片的长宽
                side_len = max(w, h)  # 样本框以原框最长边长缩放作为边长
                seed = float_num[np.random.randint(0, len(float_num))]  # 制作随机种子
                cout = 0
                if side_len * seed == 0:
                    continue
                for _ in range(5):
                    _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))  # 边长伸缩
                    _center_x = center_axes[0] + np.random.randint(int(-center_axes[0] * seed),
                                                                   int(center_axes[0] * seed))  # 中心坐标点滑动
                    _center_y = center_axes[1] + np.random.randint(int(-center_axes[1] * seed),
                                                                   int(center_axes[1] * seed))  # 中心坐标点滑动

                    _x1 = _center_x - _side_len / 2
                    _y1 = _center_y - _side_len / 2
                    _x2 = _x1 + _side_len
                    _y2 = _y1 + _side_len

                    if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:  # 当建议框超过了图片尺寸则丢弃
                        continue

                    # 计算四个坐标的偏移率
                    offset_x1 = (x1 - _x1) / _side_len
                    offset_y1 = (y1 - _y1) / _side_len
                    offset_x2 = (x2 - _x2) / _side_len
                    offset_y2 = (y2 - _y2) / _side_len

                    # 计算五个特征点偏移率
                    input_landmark = landmark[i]
                    offsets = landmarks(input_landmark, _side_len, _x1, _y1, landmarks=isLandmarks)
                    offset_px1 = offsets[0]
                    offset_py1 = offsets[1]
                    offset_px2 = offsets[2]
                    offset_py2 = offsets[3]
                    offset_px3 = offsets[4]
                    offset_py3 = offsets[5]
                    offset_px4 = offsets[6]
                    offset_py4 = offsets[7]
                    offset_px5 = offsets[8]
                    offset_py5 = offsets[9]

                    crop_box = [_x1, _y1, _x2, _y2]  # 偏移后裁剪框坐标
                    iou = utilis.IOU(crop_box, np.array(origin_box))[0]

                    if iou > 0.70:  # 正样本1
                        img_crop = img.crop(crop_box)  # 裁剪样本
                        img_resize = img_crop.resize((pics_size, pics_size))  # resize成pic_size大小
                        img_resize.save(os.path.join(positive_samples_path, "{}.jpg".format(positive_count)))  # 保存样本
                        if isLandmarks:
                            positive_target.write("positive_samples\{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} "
                                                  "{12} {13} {14} {15}\n"
                                                  .format(positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2,
                                                          offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                                          offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        else:
                            positive_target.write("positive_samples\{0}.jpg {1} {2} {3} {4} {5}\n".
                                                  format(positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                        positive_count += 1
                        positive_target.flush()

                    elif 0.4 < iou < 0.65:  # 部分样本2
                        img_crop = img.crop(crop_box)
                        img_resize = img_crop.resize((pics_size, pics_size))
                        img_resize.save(os.path.join(part_samples_path, "{}.jpg".format(part_count)))
                        if isLandmarks:
                            part_target.write("part_samples\{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} "
                                              "{12} {13} {14} {15}\n"
                                              .format(part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2,
                                                      offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                                      offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        else:
                            part_target.write("part_samples\{0}.jpg {1} {2} {3} {4} {5}\n".
                                              format(part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2))
                        part_count += 1
                        part_target.flush()

                    elif iou < 0.20:
                        img_crop = img.crop(crop_box)
                        img_resize = img_crop.resize((pics_size, pics_size))
                        img_resize.save(os.path.join(negative_samples_path, "{}.jpg".format(negative_count)))
                        if isLandmarks:
                            negative_target.write("negative_samples\{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} "
                                                  "{12} {13} {14} {15}\n"
                                                  .format(negative_count, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                        else:
                            negative_target.write("negative_samples\{0}.jpg {1} {2} {3} {4} {5}\n".
                                                  format(negative_count, 0, 0, 0, 0, 0))
                        negative_count += 1
                        part_target.flush()

                    cout = positive_count + negative_count + part_count
                    if cout % 100 == 0:
                        print("{}/{}".format(cout, stop_value))
                        print(positive_count, part_count, negative_count)

                if cout >= stop_value:
                    break

        except Exception:
            traceback.print_exc()


def landmarks(landmark, _side_len, _x1, _y1, landmarks):
    if landmarks:
        px1 = float(landmark.split()[1].strip())
        py1 = float(landmark.split()[2].strip())
        px2 = float(landmark.split()[3].strip())
        py2 = float(landmark.split()[4].strip())
        px3 = float(landmark.split()[5].strip())
        py3 = float(landmark.split()[6].strip())
        px4 = float(landmark.split()[7].strip())
        py4 = float(landmark.split()[8].strip())
        px5 = float(landmark.split()[9].strip())
        py5 = float(landmark.split()[10].strip())

        offset_px1 = (px1 - _x1) / _side_len
        offset_py1 = (py1 - _y1) / _side_len
        offset_px2 = (px2 - _x1) / _side_len
        offset_py2 = (py2 - _y1) / _side_len
        offset_px3 = (px3 - _x1) / _side_len
        offset_py3 = (py3 - _y1) / _side_len
        offset_px4 = (px4 - _x1) / _side_len
        offset_py4 = (py4 - _y1) / _side_len
        offset_px5 = (px5 - _x1) / _side_len
        offset_py5 = (py5 - _y1) / _side_len

        return offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    # generator(12, 100000)
    # generator(24, 100000)
    generator(48, 100000, isLandmarks=True)
