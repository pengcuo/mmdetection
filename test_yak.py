from mmdet.apis import init_detector, inference_detector

import glob
import sys

config_file = 'video/fast_rcnn_r50_caffe_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'video/epoch_12.pth'
device = 'cuda:0'


print("len : ", len(sys.argv))
if len(sys.argv) < 2:
    print("python test_yak.py input_path")
    exit()

input_path = sys.argv[1]
input_files = []

print("input_path : ", input_path)

model = init_detector(config_file, checkpoint_file, device=device)

for img in glob.glob(input_path + "/*jpeg"):
    print(img)
    result=inference_detector(model, img)
    model.show_result(img, result, out_file="output/" + img)
