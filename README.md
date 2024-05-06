# Model-impact
This repository contains the data and code for the following paper: 

## Introduction
This is a study on model's impact on accuracy...

## Code

-- Data_big (Datasets, the target need to start with "$<")<br>
-- batch (Batch model-based tuners) <br>
-- models (Surrogate models)<br>
-- sequential (Sequential model-based tuners)<br>
-- util (Util for tuners)<br>
-- utils (Utils for models)<br>
-- run (A simple run on system "7z", the working path is "./model-impact/code")


## Detailed reuslts
Detailed rsults...

## Datasets
The datasets are originally from 

**https://zenodo.org/records/7544891#.ZDQzsMLMLN8**:
   - Brotli
   - XGBoost
   - DConvert
   - 7z
   - ExaStencils
   - Kanzi
   - Jump3r
   - Spark
     
**https://github.com/DeepPerf/DeepPerf**:
   - LLVM
   - BDBC
   - HSQLDB
   - Polly
   - JavaGC
     
**https://zenodo.org/record/7504284#.ZDQ66sLMLN8**:
   - Lrzip
     
**https://github.com/FlashRepo/Flash-MultiConfig**:
   - noc-CM-log
   - SaC
     
**https://github.com/pooyanjamshidi/deeparch-xplorer**:
   - DeepArch
     
**https://github.com/anonymous12138/multiobj**:
   - MariaDB
     
**https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh**:
   - Polly
     
**https://github.com/xdbdilab/CM-CASL**:
   - Spark
   - Redis
   - Hadoop
   - Tomcat
     
**https://github.com/ai-se/BEETLE**:
   - Storm

Details of the datasets are given in our paper.

## Raw experiments results

The experiment data reported in the work can be found at: https://xxx/xxx. <br>
The naming rule follow as: <br>
"PickleLocker\_[tuner]\_[models\results]/Data\_[big\small]/[Data]/\model\_[seed+num].csv"  <br>
"PickleLocker\_[tuner]\_[models\results]/Data\_[big\small]/[Data]/\model\_[seed+num]\_[step+num].p"  <br>
- e.g. Result: "./PickleLocker_atconf_results/Data_big/7z/RF_seed101.csv"; 
- e.g. Model: "./PickleLocker_atconf_models/Data_big/7z/GP_seed101_step20.p"

## Supplementary
supplementary.pdf contains the specific best and worst models in this study.
