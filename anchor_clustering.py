# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:21:58 2019

clustering for getting optimized anchors

methods:
    k-means

@author: Wen Wen
"""
from sklearn.cluster import KMeans
import numpy as np
import time
import os
import json
import argparse
import math
import matplotlib.pyplot as plt


def load_json_annotations(filepath, jsonlabel):
    annotationdict={}
    folderdict=os.listdir(filepath)
    for foldername in folderdict:
        jsonpath=os.path.join(filepath,foldername)
        if '.' in jsonpath:
            continue
        # load the json files
        jsondict=os.listdir(jsonpath)
        for jsonname in jsondict:
            if jsonlabel in jsonname and '.json'in jsonname:
                annotationdict[foldername]={}
                annotationdict[foldername]=json.load(open(os.path.join(jsonpath,jsonname)))
    
    return annotationdict

def returnBoxArea(box):
    # return area of box
    return box[0]**2+box[1]**2

def get_bboxes(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    bboxlist=[]
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            if len(annotationdict[foldername][imagename])==0:
                continue
            else:
                for anno in annotationdict[foldername][imagename]['annotations']:
                    if anno['width'] and anno['height']:
                        bboxlist.append([round(anno['width']*0.46875),round(anno['height']*0.625)])
                
    return bboxlist

def groupBBox(bboxlist,threshlist):
    """
    group bbox into different groups by grid size, save all the bbox into a dict
    that contain multiple bbox lists
    
    """
    xlist=bboxlist.copy()
    xlist.sort(key=returnBoxArea,reverse=True)
    bboxdict={}
    print('grouping bbox based on threshlist...')
    for thresh in threshlist:
        curlist=[]
        for i in range(len(xlist)):
            box=xlist.pop()
            if box[0]**2+box[1]**2<thresh**2:
                curlist.append(box)
            else:
                break
    
        bboxdict[thresh]=curlist
    return bboxdict

def simpleKMeans(xlist,n_cluster=2,random_state=0,algorithm='auto'):
    """
    
    output:
        centroids: centroids acquired from clustering
        labels: clustering result of input array
        score: sum of distances from centroids to all the points in their cluster
        time: processing time of current k-means clustering
        
    """
    if len(xlist)==0:#skip empty list
        return [],[],1,0
    
    starttime=time.time()
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state,algorithm=algorithm).fit(xlist)
    processtime = time.time()-starttime
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = -kmeans.score(xlist)
    
    return centroids, labels, score, processtime

def adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.8):
    """
    take in a bboxlist and perform k-means clustering where k ranges from 2 to
    a maximum (defalut is 10). 
    Using Elbow method to get the optimized k value for the bboxlist, where the
    elbow point is defined by a score threshold. when current score is larger
    than last score, i.e. score>scorethresh*last_score, the last point is the
    elbow point. note that the changing threshold will have huge impact to the 
    final result.
    
    """
    clusterlist=[]
    for thresh in bboxdict:
        grouplist=[]
        lastscore=float('Inf')
        for n_cluster in range(2,max_num_center):
            # k-means clustering        
            centroids, labels, score, processtime = simpleKMeans(xlist=bboxdict[thresh], n_cluster=n_cluster)
            print('clustering for k={}, processing time={}, score={}'.format(n_cluster,processtime,score))
            singledict={}
            singledict['k']=n_cluster
            singledict['centroids']=centroids.tolist()
            singledict['labels']=labels.tolist()
            singledict['score']=score
            singledict['processtime']=processtime
            grouplist.append(singledict)
            if score >= lastscore*scorethresh:
                print('current score is {}% of last score, stop clustering for current layer'.format(score/lastscore))
                break
            lastscore=score
        
        clusterlist.append({'thresh':thresh, 'final_centro':singledict['centroids']})
    return clusterlist

def plotScatter(bboxdict, clusterdict, savepath=''):
    chartaxis = [0.0,310.0,0.0,310.0]
    plt.figure(figsize=(6,6),dpi=100)
    plt.axis(chartaxis)
    colormap=['#33ccff', '#00ff00', '#33ccff','#00ff00', '#33ccff', '#00ff00']# use html colors
    centrolist=[i['final_centro'] for i in clusterdict]

    ax = plt.subplot()
    
    # plot scatter for benchmarks
    for i, thresh in zip(range(len(bboxdict)),bboxdict):
        x_box = [i[0] for i in bboxdict[thresh]]
        y_box = [i[1] for i in bboxdict[thresh]]
        area_box = np.ones(len(x_box))*1
        ax.scatter(x_box, y_box, s=area_box, alpha=0.2, c=colormap[i], marker='o')# colormap[i]*len(x_box)
    

    # plot centroids
    for anchorlist in centrolist:
        x_anchor = [i[0] for i in anchorlist]
        y_anchor = [i[1] for i in anchorlist]
        area_anchor = np.ones(len(anchorlist))*100
        ax.scatter(x_anchor, y_anchor, s=area_anchor, alpha=1, c='r', marker='.')
    
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.title('Distribution of Groundtruth BBox and anchors')
    plt.grid(True)
    plt.savefig(os.path.join(savepath,'scatter.png'),dpi=1000)
    print('scatter saved as\n {}'.format(os.path.join(savepath,'scatter.png')))
    plt.show()
    plt.close()
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset/', 
                        help="File path of input data")
    #D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    
    parser.add_argument('--json_label', type=str, 
                        default='annotation', 
                        help="label to specify json file")
    #val_VIVA_format_crop.json
    
    parser.add_argument('--save_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset/ssd cluster result', 
                        help="path to save")
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    savepath = args.save_path
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get bboxlist from annotation
    bboxlist = get_bboxes(annotationdict)
    
    # setup threshlist, where threshold is diagnal length in pixel
    # to get the threshlist easily, check the get_anchor_from_bench_mark.py
    # run the file and check h_pel_list
    
    '''threshes for various dataset'''
    #threshlist = [15.78,30,60,100,150,300] # Threshlist for BDD dataset, of grid size [19,10,5,3,2,1]
    threshlist=[13.63, 17.64, 33.33, 75, 150, 300] # Threshlist for Caltech dataset, of grid size [22,17,9,4]
    
    bboxdict = groupBBox(bboxlist,threshlist)
    
    # perform k-means for all the layers
    clusterdict = adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.75)
    json.dump(clusterdict, open(os.path.join(savepath,'k-means_cluster_result_ssd.json'), 'w'),sort_keys=True, indent=4)

    # draw figures of [sum of error over k values]
    plotScatter(bboxdict, clusterdict, savepath=savepath)




"""End of file"""