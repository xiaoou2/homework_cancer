from judge.ultralytics import YOLO

# model = YOLO(r"runs/detect/train/weights/last.pt").train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})


import os
path = r'runs/detect/result1/labels/'

data = os.listdir(path)
for i in data:
    i = path+i
    os.remove(i)

# print(torch.cuda.is_available())
# #加载模型
model = YOLO(r"runs/detect/train/weights/last.pt")  # or a segmentation model .i.e yolov8n-seg.pt


model.track(
    #输入视频路径
    source=r"C:\Users\Administrator\Desktop\t\breast_mp4\pharyngolaryngeal.mp4",
    # stream=True,
    tracker="bytetrack.yaml",  # or 'bytetrack.yaml'
    save = True,
    name = "result1",
    conf = 0.6,
    save_txt = True
)

path = r'runs/detect/result1/labels/'
data = os.listdir(path)
num = []
area = []
X = len(data)
file =''
for i in data:
    i = path+i
    file = open(i,'r')
    p = file.readlines()
    num_flag = 0
    position = []
    max_area = 0.00
    for k in p:
        k = k.split('\n')[0]
        list = k.split(" ")
        temp = float(list[3])*float(list[4])
        if temp>max_area:
            temp = round(temp,2)
            max_area = temp
        num_flag+=1
    area.append(max_area)
    num.append(num_flag)
print(X)
print(num)
print(area)
file.close()
