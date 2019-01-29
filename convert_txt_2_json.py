# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 11:41:56 2019

convert txt Groundtruth to tf-obj-api json files
this is for UAVDT dataset

the format of each row in txt file is like:
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
<out-of-view>,<occlusion>,<object_category>
for class-index pair of object category, car:1,truck:2,bus:3

@author: Wen Wen
"""
import numpy as np
import argparse
import os
import json

def getClassFromID(classid):
    iddict={1:'car',2:'truck',3:'bus'}
    return iddict[classid]

def convertAnnotationVIVA(filename,gt_array,annodict,filelabel):
    """
    update the whole annodict with new gt_array, note that the image name should
    be as FOLDER/IMGNAME.jpg
    
    """
    folder=filename.replace(filelabel,'')
    newdict=annodict
    dtype=[('index',int),('id',int),('left',int),('top',int),('w',int),
           ('h',int),('out-of-view',int),('occlusion',int),('class',int)]
    gt=[tuple(i) for i in gt_array]
    gt=np.array(gt,dtype=dtype)
    gt=np.sort(gt,order=['index','id'])
    
    """
    now gt is fielded ndarray, 
    using example as: gt['left'] to get left value of row 5
    value
    
    """
    lastid=0
    for row in gt:
        currentid=int(row['index'])
        if currentid>lastid:
            newanno={}
            newanno['annotations']=[]
            imgname=folder+("/img%06d.jpg" % currentid)
            newanno['name']=imgname
            newanno['width']=1024
            newanno['height']=540
        newbox={}
        newbox['label']=getClassFromID(row['class'])
        newbox['id']=int(row['id'])
        newbox['shape']=["Box", 1]
        newbox['category']='sideways'
        newbox['x']=int(row['left'])
        newbox['y']=int(row['top'])
        newbox['width']=int(row['w'])
        newbox['height']=int(row['h'])
        
        # append current bbx to the image anno
        newanno['annotations'].append(newbox)
              
        newdict[imgname]=newanno
        # update the id
        lastid=currentid
        
    return newdict

def divideList(jsonlist,trainlabel,testlabel):
    trainlist=[]
    testlist=[]
    trainlabel=[i.split('_')[0] for i in trainlabel]
    testlabel=[i.split('_')[0] for i in testlabel]
    
    for i in jsonlist:
        if 'whole' in i and i.split('_')[0] in trainlabel:
            trainlist.append(i)
        elif 'whole' in i and i.split('_')[0] in testlabel:
            testlist.append(i)
    
    return trainlist, testlist

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 winter/UAV project/GT', 
        help="select the file path for txt groundtruth")
    parser.add_argument('--train_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 winter/UAV project/M_attr/train', 
        help="select the file path for training list")
    parser.add_argument('--test_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 winter/UAV project/M_attr/test', 
        help="select the file path for testing list")
    parser.add_argument('--file_label',type=str,default='_gt_whole.txt',
                        help='select the label for the txt ground truth that to be convert,default=_gt_whole.txt')
    parser.add_argument('--save_name',type=str,default='UAV-benchmark-M-VIVA.json',
                        help='the saving name for the new groundtruth file')
    args = parser.parse_args()
    filepath=args.file_path
    trainpath=args.train_path
    testpath=args.test_path
    filelabel=args.file_label
    savename=args.save_name
    
    jsonlist=os.listdir(filepath)
    trainlabel=os.listdir(trainpath)
    testlabel=os.listdir(testpath)
    trainlist, testlist = divideList(jsonlist,trainlabel,testlabel)
    
   
    # create training benchmark
    annodict={}
    for filename in trainlist:
        if filelabel not in filename:
            continue
        gt_array=np.loadtxt(os.path.join(filepath,filename),delimiter=',').astype(np.float32)
        annodict=convertAnnotationVIVA(filename,gt_array,annodict,filelabel)

    with open(os.path.join(filepath,savename.replace('.json','-train.json')),'w') as fp:
        json.dump(annodict,fp,sort_keys=True, indent=4)

    # create testing benchmark
    annodict={}
    for filename in testlist:
        if filelabel not in filename:
            continue
        gt_array=np.loadtxt(os.path.join(filepath,filename),delimiter=',').astype(np.float32)
        annodict=convertAnnotationVIVA(filename,gt_array,annodict,filelabel)

    with open(os.path.join(filepath,savename.replace('.json','-test.json')),'w') as fp:
        json.dump(annodict,fp,sort_keys=True, indent=4)

    
"""End of file"""