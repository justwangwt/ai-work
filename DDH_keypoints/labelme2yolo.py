import os 
import json
import shutil
import numpy as np 
from tqdm import tqdm

Dataset_root='DDH_keypoint'

Dataset_DDH=os.path.join(Dataset_root,'dataset','DDH')
Dataset_normal=os.path.join(Dataset_root,'dataset','normal')

bbox_class={
    'skeleton':0
}

keypoint_class=['TeardropR','TeardropL','TiR','TiL','FHR','FHL',
                'tonnisR1','tonnisR2','tonnisL1','tonnisL2']

os.chdir(Dataset_DDH)
#os.mkdir('labels_txt')


def pross_single_json(labelme_path,save_folder='../../labels'):
    
    with open(labelme_path,'r',encoding='utf-8')as f:
        labelme=json.load(f)
        
    img_width=labelme['imageWidth']#图像宽度
    img_height=labelme['imageHeight']#图像高度
    
    #生成yolo格式的txt文件
    #suffix=labelme_path.split('.')[-2]
    suffix=labelme_path.split('.json')[0]
    yolo_txt_path=suffix+'.txt'
    rec_num=0
    with open(yolo_txt_path,'w',encoding='utf-8')as f:
        
        for each_ann in labelme['shapes']:#遍历每个标注
            
            if each_ann['shape_type']=='rectangle':
                rec_num+=1
                
        if rec_num>=1:
            for each_ann in labelme['shapes']:
                
                if each_ann['shape_type']=='rectangle':#每个框，在txt里写一行
                    
                    yolo_str=''
                    
                    ##框的信息
                    #框的类别
                    bbox_class_id=bbox_class[each_ann['label']]
                    yolo_str+='{} '.format(bbox_class_id)
                    #左上角和右下角的XY像素坐标
                    bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                    bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                    bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                    #框中心点的XY像素坐标
                    bbox_center_x=int((bbox_top_left_x+bbox_bottom_right_x)/2)
                    bbox_center_y=int((bbox_top_left_y+bbox_bottom_right_y)/2)
                    #框宽度
                    bbox_width=bbox_bottom_right_x-bbox_top_left_x
                    #框高度
                    bbox_height=bbox_bottom_right_y-bbox_top_left_y
                    #框中心点归一化坐标
                    bbox_center_x_norm=bbox_center_x/img_width
                    bbox_center_y_norm=bbox_center_y/img_height
                    #框归一化宽度
                    bbox_width_norm=bbox_width/img_width
                    #框归一化高度
                    bbox_height_norm=bbox_height/img_height
                    
                    yolo_str+='{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm,bbox_center_y_norm,bbox_width_norm,bbox_height_norm)
                    
                    #找到该框中所有的关键点，存在字典 bbox_keypoints_dict 中
                    bbox_keypoints_dict={}
                    for each_ann in labelme['shapes']:#遍历所有标注
                        if each_ann['shape_type']=='point':
                            #关键点XY坐标、类别
                            x=int(each_ann['points'][0][0])
                            y=int(each_ann['points'][0][1])
                            label=each_ann['label']
                            if (x>bbox_top_left_x)&(x<bbox_bottom_right_x)&(y<bbox_bottom_right_y)&(y>bbox_top_left_y):
                                bbox_keypoints_dict[label]=[x,y]
                            
                        elif each_ann['shape_type']=='circle':
                            x=int(each_ann['points'][0][0])
                            y=int(each_ann['points'][0][1])
                            label=each_ann['label']
                            if (x>bbox_top_left_x)&(x<bbox_bottom_right_x)&(y<bbox_bottom_right_y)&(y>bbox_top_left_y):
                                bbox_keypoints_dict[label]=[x,y]
                    
                    ##把关键点按顺序排好
                    for each_class in keypoint_class:#遍历每一类关键点
                        if each_class in bbox_keypoints_dict:
                            keypoint_x_norm=bbox_keypoints_dict[each_class][0]/img_width
                            keypoint_y_norm=bbox_keypoints_dict[each_class][1]/img_height
                            yolo_str+='{:.5f} {:.5f} {} '.format(keypoint_x_norm,keypoint_y_norm,2)
                        else:
                            yolo_str+='0 0 0 '
                    #写入txt文件中
                    f.write(yolo_str+'\n')
                    
        else:#如果没框，手动设框
            yolo_str=''
            bbox_class_id=0
            yolo_str+='{} '.format(bbox_class_id)
            #左上角和右下角的XY像素坐标
            bbox_top_left_x =41.391304347826086
            bbox_bottom_right_x=965.3043478260869
            bbox_top_left_y=218.17391304347825
            bbox_bottom_right_y=876.8695652173913
            #框中心点的XY像素坐标
            bbox_center_x=int((bbox_top_left_x+bbox_bottom_right_x)/2)
            bbox_center_y=int((bbox_top_left_y+bbox_bottom_right_y)/2)
            #框宽度
            bbox_width=bbox_bottom_right_x-bbox_top_left_x
            #框高度
            bbox_height=bbox_bottom_right_y-bbox_top_left_y
            #框中心点归一化坐标
            bbox_center_x_norm=bbox_center_x/img_width
            bbox_center_y_norm=bbox_center_y/img_height
            #框归一化宽度
            bbox_width_norm=bbox_width/img_width
            #框归一化高度
            bbox_height_norm=bbox_height/img_height
            
            yolo_str+='{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm,bbox_center_y_norm,bbox_width_norm,bbox_height_norm)
            
            #找到该框中所有的关键点，存在字典 bbox_keypoints_dict 中
            bbox_keypoints_dict={}
            for each_ann in labelme['shapes']:#遍历所有标注
                if each_ann['shape_type']=='point':
                    #关键点XY坐标、类别
                    x=int(each_ann['points'][0][0])
                    y=int(each_ann['points'][0][1])
                    label=each_ann['label']
                    if (x>bbox_top_left_x)&(x<bbox_bottom_right_x)&(y<bbox_bottom_right_y)&(y>bbox_top_left_y):
                        bbox_keypoints_dict[label]=[x,y]
                
                elif each_ann['shape_type']=='circle':
                            x=int(each_ann['points'][0][0])
                            y=int(each_ann['points'][0][1])
                            label=each_ann['label']
                            if (x>bbox_top_left_x)&(x<bbox_bottom_right_x)&(y<bbox_bottom_right_y)&(y>bbox_top_left_y):
                                bbox_keypoints_dict[label]=[x,y]
            
            ##把关键点按顺序排好
            for each_class in keypoint_class:#遍历每一类关键点
                if each_class in bbox_keypoints_dict:
                    keypoint_x_norm=bbox_keypoints_dict[each_class][0]/img_width
                    keypoint_y_norm=bbox_keypoints_dict[each_class][1]/img_height
                    yolo_str+='{:.5f} {:.5f} {} '.format(keypoint_x_norm,keypoint_y_norm,2)
                else:
                    yolo_str+='0 0 0 '
            #写入txt文件中
            f.write(yolo_str+'\n')
            
    shutil.move(yolo_txt_path,save_folder)
    print('{}-->{}转换完成'.format(labelme_path,yolo_txt_path))

os.chdir('labels')

save_folder='../../labels'
for labelme_path in os.listdir():
    pross_single_json(labelme_path,save_folder=save_folder)
print('yolo格式的txt标注文件以保存至',save_folder)