
export PYTHONPATH=$PWD:$PWD/cu11_deps

#config=/home/pengcuo.zeren/work/mmdetection/configs/fast_rcnn/fast_rcnn_r50_caffe_fpn_1x_coco.py
config=/home/pengcuo.zeren/work/mmdetection/tutorial_exps/fast_rcnn_r50_caffe_fpn_1x_coco.py

/home/pengcuo.zeren/.local/torchrun --nproc_per_node=1 --nnodes=1 tools/train.py $config --launcher pytorch --no-validate
