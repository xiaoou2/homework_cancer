import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from django.http import HttpResponse
from numpy import reshape

from .bts.classifier import BrainTumorClassifier
from .bts.model import DynamicUNet


def get_arguments():
    """Returns the command line arguments as a dict"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False, type=str,default=r'3000.png',
                        help='Single input file name.')
    parser.add_argument('--dir', required=False,
                        type=str, help='Directory name with input images')
    parser.add_argument('--ofp', required=False,default='output',
                        type=str, help='Single output file path with name.Use this if using "file" flag.')
    parser.add_argument('--odp', required=False,default='output',
                        type=str, help='Directory path for output images.Use this if using "dir" flag.')
    args = parser.parse_args()
    args = {'file': args.file, 'folder': args.dir,
            'ofp': args.ofp, 'odp': args.odp}
    return args


class Api:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.segment = None
        self.contours_img = None
        self.input_x = None
        self.input_y = None
    def call(self, file, folder=None, ofp=r"E:\Cancer\production\ai\judge\judgeMain\output", odp=None):
        """Method saves the predicted image by taking different parameters."""
        if file != None and folder != None:
            print('"folder" flag and "file" flag cant be used together')
            return
        startTime = int(round(time.time() * 1000))
        model = self._load_model()
        # For a single file
        if file != None:
            image = self._get_file(file)
            output = self._get_model_output(image, model)
            name, extension = file.split('.')
            img1 = name.split('\\')[-1]+'_predicted'+'.'+extension
            img2 = name.split('\\')[-1]+'_predicted' + 'segmentation.jpg'
            img3 = name.split('\\')[-1]+'_predicted'+'contours_img.jpg'
            save_path = ''
            if ofp:
                save_path = os.path.join(ofp,img1)
            image_array = np.array(output)
            for k in range(image_array.shape[0]):
                for p in range(image_array.shape[1]):
                    if image_array[k][p] == 0:
                        self.segment[k][p][0] = 0
                        self.segment[k][p][1] = 255
                        self.segment[k][p][2] = 0

            gray_img = image_array.astype(np.uint8)
            ret, mask = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
            temp = mask.astype('uint8')
            contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = cv2.drawContours(self.contours_img, contours, -1, (0, 0, 255), 1)

            cv2.imwrite(os.path.join(ofp,img3), contour_img)
            cv2.imwrite(os.path.join(ofp, img2), self.segment)
            self._save_image(output, save_path)
            print(f'Output Image Saved At {save_path}')
            Area = ''
            PERIMETER = ''
            LEFT = ''
            RIGHT = ''
            TOP = ''
            BOTTOM = ''

            for cnt in contours:
                # cnt = contours[0]
                area = cv2.contourArea(cnt)
                area = round(area, 3)
                perimeter = cv2.arcLength(cnt, True)
                perimeter = round(perimeter, 3)
                left = tuple(cnt[cnt[:, :, 0].argmin()][0])
                right = tuple(cnt[cnt[:, :, 0].argmax()][0])
                top = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

                Area = Area + str(area)
                PERIMETER = PERIMETER + str(perimeter)
                LEFT = LEFT + str(left)
                RIGHT = RIGHT + str(right)
                TOP = TOP + str(top)
                BOTTOM = BOTTOM + str(bottom)

            # plt.imshow(contour_img)
            # plt.show()
            MASK = mask.astype('uint8')
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(MASK,
                                                                                connectivity=4)  # connectivity参数的默认值为8
            stats = stats[stats[:, 4].argsort()]
            boxs = stats[:-1]

            LEFTTOP_POINT = ''
            RIGHTDOWN_POINT = ''
            CENTER_POINT = ''
            WIDTH = ''
            HEIGHT = ''
            for b in boxs:
                x0, y0 = b[0], b[1]
                x1 = b[0] + b[2]
                y1 = b[1] + b[3]
                width = b[2]
                height = b[3]
                start_point, end_point = (x0, y0), (x1, y1)
                LEFTTOP_POINT = LEFTTOP_POINT + '(' + str(start_point[0]) + ',' + str(start_point[1]) + ')'
                RIGHTDOWN_POINT = RIGHTDOWN_POINT + '(' + str(end_point[0]) + ',' + str(end_point[1]) + ')'
                CENTER_POINT = CENTER_POINT + '(' + str((start_point[0] + end_point[0]) / 2.0) + ',' + str(
                    (start_point[1] + end_point[1]) / 2.0) + ')'
                WIDTH = WIDTH + str(width)
                HEIGHT = HEIGHT + str(height)

            print('图像分辨率 : (%d,%d)' % (self.input_x, self.input_y))
            print('左上角坐标 : ' + LEFTTOP_POINT)
            print('右下角坐标 : ' + RIGHTDOWN_POINT)
            print('中心坐标 ： ' + CENTER_POINT)
            print('区域宽度 : ' + WIDTH)
            print('区域高度 : ' + HEIGHT)
            print('区域面积 ： ' + Area)
            print('区域周长 : ' + PERIMETER)
            print('左极点坐标 : ' + LEFT)
            print('右极点坐标 : ' + RIGHT)
            print('上极点坐标 : ' + TOP)
            print('下极点坐标 : ' + BOTTOM)
            info = {
                '图像分辨率': '(' + str(self.input_x) + ',' + str(self.input_y) + ')',
                '左上角坐标': LEFTTOP_POINT,
                '右下角坐标': RIGHTDOWN_POINT,
                '中心坐标': CENTER_POINT,
                '区域宽度': WIDTH,
                '区域高度': HEIGHT,
                '区域面积': Area,
                '区域周长': PERIMETER,
                '左极点坐标': LEFT,
                '右极点坐标': RIGHT,
                '上极点坐标': TOP,
                '下极点坐标': BOTTOM,
            }
            endTime = int(round(time.time() * 1000))
            detalTime = endTime - startTime
            result = {"message": 'success', "code": '200',
                      "img1": 'http://127.0.0.1:8000/static/' + img1,
                      "img2": 'http://127.0.0.1:8000/static/' + img2,
                      "img3": 'http://127.0.0.1:8000/static/' + img3,
                      "detalTime": detalTime,
                      "info": info,
                      "time": endTime
                      }

            return result
        elif folder != None:
            image_list = os.listdir(folder)
            for file in image_list:
                file_name = os.path.join(folder, file)
                image = self._get_file(file_name)
                output = self._get_model_output(image, model)

                name, extension = file.split('.')
                save_path = name+'_predicted'+'.'+extension

                save_path = os.path.join(
                    odp, save_path) if odp else os.path.join(folder, save_path)
                self._save_image(output, save_path)
                print(f'Output Image Saved At {save_path}')


    def _load_model(self):
        """Load the saved model and return it."""
        filter_list = [16, 32, 64, 128, 256]

        model = DynamicUNet(filter_list).to(self.device)
        classifier = BrainTumorClassifier(model, self.device)
        model_path = r'E:\Cancer\production\ai\judge\judgeMain\na\saved_models\UNet-[16, 32, 64, 128, 256].pt'
        classifier.restore_model(model_path)
        print(
            f'Saved model at location "{model_path}" loaded on {self.device}')
        return model

    def _get_model_output(self, image, model):
        """Returns the saved model output"""
        image = image.view((-1, 1, 512, 512)).to(self.device)
        output = model(image).detach().cpu()
        output = (output > 0.5)
        output = output.numpy()
        output = np.resize((output * 255), (512, 512))
        return output

    def _save_image(self, image, path):
        """Save the image to storage specified by path"""
        image = Image.fromarray(np.uint8(image), 'L')
        image.save(path)

    def _get_file(self, file_name):
        """Load the image by taking file name as input"""
        default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])

        img = Image.open(file_name)
        self.segment = cv2.imread(file_name).copy()
        self.contours_img = cv2.imread(file_name).copy()
        self.input_x = self.segment.shape[0]
        self.input_y = self.segment.shape[1]
        image = default_transformation(img)
        return TF.to_tensor(image)


if __name__ == "__main__":
    print(666)
    args = get_arguments()
    api = Api()
    api.call(**args)

def api2(file):
    print(file)
    api = Api()
    res=api.call(file)
    return res
