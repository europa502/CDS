
# Content Detection System


This program is made to detect specific contents from video files. Currently it just detects objects from video frames only.


## Prerequisites- 

1. Python 2.7
2. Python libraries - Tensorflow, OpenCV.
3. Video file for testing the model.

### 1. Collect dataset- 
  Download images that represent each category. At least two categories are required to train your model initially. More
  the images, the better. But make sure that only relevent images with charecterstic features are present. Otherwise it may reduce 
  the accureacy of the model.
  For this purpose you can use batch downloader extentions for your browsers.
  Sort these images into respective folders.
  ![screenshot from 2019-03-03 09-35-39](https://user-images.githubusercontent.com/26405791/53693395-ca9aa800-3d97-11e9-86fd-7902ad50355b.png)


### 2. Train your model-
  Run trainer.py to train develop your model.
  ```
  root@europa:~# python trainer.py --bottleneck_dir bottlenecks/ --image_dir training_set/ --output_labels CDS_output_labels.txt 
  --output_graph CDS_output_graph.pb
  ```
  
### 3. Test your model
  Run cds.py
  ```
  root@europa:~# python cds.py --video vid.mp4 --intensity 100 --verbosity 2 --sensitivity 0.8 --graph CDS_output_graph.pb 
  --labels CDS_output_labels.txt
  ```
### 4. Results-

If it works perfectly then you should get something like this

```
root@europa:~# python cds.py --video vid.mp4 --intensity 100 --verbosity 2 --sensitivity 0.8 --graph CDS_output_graph.pb 
  --labels CDS_output_labels.txt
Successfully created the directory blacklist/
Width:		1280.0
Height:		720.0
FPS:		24.0
Total Frames:	9611.0
Length:		400.458333333 sec
current frame 0
current frame 100
current frame 200
current frame 300
current frame 400
.
.
.
current frame 9400
current frame 9500
current frame 9600
.....................................................
helmets : 1.8386348802086383 %
pistols : 0.7111849018694115 %
rifles : 0.6209323493484098 %
woman : 11.68438792073656 %
revolvers : 0.45700693417348953 %
man : 85.61387554240541 %
-----------------------------------------------------
Total time taken:	19.4948489666
```

## 5. Retraining your model- 
 You may need to retrain your model to improve its accuracy. You can therefore use the trainer.py to retrain your model with more dataset.
 ```
 root@europa:~# python trainer.py --bottleneck_dir bottlenecks/ --image_dir training_set/ --output_labels CDS_output_labels.txt 
  --output_graph CDS_output_graph.pb --how_many_training_steps 20000
  ```
  
 
