1.安装yolov8
pip install ultralytics --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

2.验证安装成功
import ultralytics
ultralytics.checks()

3.安装其他第三方工具包
pip install numpy opencv-python pillow pandas matplotlib seaborn tqdm wandb seedir emoji -i https://pypi.tuna.tsinghua.edu.cn/simple

4.登录可视化工具wandb
terminal：wandb login
复制Api key 登录

5.yolo8训练参数介绍

## YOLOV8-关键点检测预训练模型

yolov8n-pose.pt

yolov8s-pose.pt

yolov8m-pose.pt

yolov8l-pose.pt

yolov8x-pose.pt

yolov8x-pose-p6.pt

## 几个比较重要的训练参数

model YOLOV8模型

data 配置文件（.yaml格式）

pretrained 是否在预训练模型权重基础上迁移学习泛化微调

epochs 训练轮次，默认100

batch batch-size，默认16

imgsz 输入图像宽高尺寸，默认640

device 计算设备（device=0 或 device=0,1,2,3 或 device=cpu）

project 项目名称，建议同一个数据集取同一个项目名称

name 实验名称，建议每一次训练对应一个实验名称

optimizer 梯度下降优化器，默认'SGD'，备选：['SGD', 'Adam', 'AdamW', 'RMSProp']

close_mosaic 是否关闭马赛克图像扩增，默认为0，也就是开启马赛克图像扩增

cls 目标检测分类损失函数cls_loss权重，默认0.5

box 目标检测框定位损失函数box_loss权重，默认7.5


dfl 类别不均衡时Dual Focal Loss损失函数dfl_loss权重，默认1.5。

pose 关键点定位损失函数pose_loss权重，默认12.0（只在关键点检测训练时用到）

kobj 关键点置信度损失函数keypoint_loss权重，默认2.0（只在关键点检测训练时用到）

如果你的数据集和MS COCO数据集的图像域**类似**（街景、动植物、生活用品），可以保留预训练模型权重，在自己的数据集上迁移学习微调分类输出层或所有层。站在巨人的肩膀上，复用预训练模型在MS COCO数据集上学习到的图像特征。（Transfer Learning, Fine Tuning）

如果你的数据集和MS COCO数据集的图像域**不类似**（医疗影像、显微镜图像、工业检测、天文照片、动画、油画），可以随机初始化模型权重，在自己的数据集上重新训练所有层。（From Scratch）。或者冻结底层权重，只重新训练顶层，复用预训练模型在MS COCO数据集上学习到的底层图像特征。

****************！！！！！！！！！！！！！！*******************

6.训练关键点检测模型
terminal：
yolo pose train data=DDH_keypoint.yaml model=yolov8n-pose.pt imgsz=1024 project=DDH_keypoint name=DDH batch=16 device=0

************************************************************
7.文件结构解析
yolov8n-pose.pt/yolov8n.pt yolov8模型

DDH_keypoints.yaml 模型训练配置文件

datasets 数据集文件

DDH_keypoint(dir) 训练产生的工程文件包括（评估指标图和训练出的预测模型）

labelme_datashow.py 将labelme标注的数据信息通过excel显示出来

labelme2yolo.py 将labelme的标注信息格式转换成yolo可识别的形式

8.训练日志和评估指标可视化
## 训练得到的模型权重文件

最优模型：`Project_Name/Name/weights/best.pt`

最终模型：`Project_Name/Name/weights/last.pt`

## 数据集标注统计

目标检测框的中心点位置分布、宽高分布：`labels.jpg`

目标检测框的中心点X、中心点Y、宽、高相关分布：`labels_correlogram.jpg`

# 训练集：某一个batch的标注可视化

`train_batch0.jpg`

`train_batch1.jpg`

`train_batch2.jpg`

# 测试集：某一个batch的标注、预测结果可视化

标注：`val_batch0_labels.jpg`

预测结果：`val_batch0_pred.jpg`

标注：`val_batch1_labels.jpg`

预测结果：`val_batch1_pred.jpg`

# 目标检测评估指标

不同置信度的Precision：`BoxP_curve.png`

不同置信度的Recall：`BoxR_curve.png`

不同置信度的PR曲线：`BoxPR_curve.png`

不同置信度的F1：`BoxF1_curve.png`

目标检测框混淆矩阵：`confusion_matrix.png`

## 关键点检测评估指标

不同置信度的Precision：`PoseP_curve.png`

不同置信度的Recall：`PoseR_curve.png`

不同置信度的PR曲线：`PosePR_curve.png`

不同置信度的F1：`PoseF1_curve.png`

## 训练日志

训练过程中的损失函数、测试集评估指标：`results.csv`、`results.png`