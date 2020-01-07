# DPU CV Application

**Author: Qizhang Li**

This is the application for face recognization using CNN with Xilinx DNNDK. The purpose of the project is trying to find out and recongize the specified person in the video stream. This will be running at ZCU102 SoC board with Xilinx DPU and a IMX274 FMC board.

processing the data in this format:
-training_data:
    -me:
        -me.1.jpg
    -others
        -others.1.jpg
-testing_data:
    -me:
        -me.2.jpg
    -others
        -others.2.jpg

running with command  python train.py to get the model
use predict.py to validation
use frozen.sh, quan.sh and dnnc.sh to generate elf file(in src)

It is a software part of the whole project https://github.com/neu-ece-eece4534-sp19/f19-project-yu-deng-li.


## Dependencies

- tensorflow
- opencv





## License
The training dataset is based on Labeled Faces in the Wild Home.http://vis-www.cs.umass.edu/lfw/#resources
