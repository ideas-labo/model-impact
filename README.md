# Model-impact
This repository contains the data and code for the following paper: 
> Accuracy can Lie: On the Impact of Surrogate Model in Configuration Tuning

## Introduction
To ease the expensive measurements during configuration tuning, it is natural to build a surrogate model as the replacement of the system and thereby the configuration performance can be cheaply evaluated. Yet, a stereotype therein is that the higher the model accuracy, the better the tuning result would be, or vice versa. This 'accuracy is all' belief drives our research community to build more and more accurate models and criticize a tuner due to the inaccuracy of the model it uses. However, this practice raises some previously unaddressed questions, e.g., whether the model and its accuracy are really that important for the tuning result? Do those somewhat small accuracy improvements reported (e.g., a few \% error reduction) in existing work really matter much to the tuners? What role does model accuracy play in the impact of tuning quality? To answer those related questions, in this paper, we conduct one of the largest-scale empirical studies to date---running over the period of 13 months $24\times7$---that covers 10 models, 17 tuners, and 29 systems from the existing works while under four different commonly used metrics, leading to ? cases of investigation. Surprisingly, our key findings reveal that the accuracy can lie: there are a considerable number of cases where higher accuracy actually leads to no improvement in the tuning outcomes, or even worse, it can degrade the tuning quality. We also discover that the chosen models in most proposed tuners are sub-optimal and that the required \% of accuracy change to significantly improve tuning quality varies according to the range of model accuracy. From those, we provide in-depth discussions of the rationale behind and offer several lessons learned as well as insights for future opportunities. Most importantly, this work poses a clear message to the community that we should take one step back from the natural 'accuracy is all' belief for model-based configuration tuning.

## Code

-- Data_big (Datasets, the target need to start with "$<")<br>
-- batch (Batch model-based tuners) <br>
-- models (Surrogate models)<br>
-- sequential (Sequential model-based tuners)<br>
-- util (Util for tuners)<br>
-- utils (Utils for models)<br>
-- requirements.txt (Essential requirments need to be installed) <br>
-- run (A simple run on system "7z", the working path is "./model-impact/code")

To run the code, cd "./model-impact/code" as working path and install the essential requirements: 
'''
pip install -r requirements.txt,  
'''
And you can run below code to have a quick start:
'''
python3 run.py
'''


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
And the naming rule follow as: <br>
"PickleLocker\_[tuner]\_results/Data\_[big\small]/[Data]/[model]\_[seed+num].csv"  <br>
- e.g. Result: "./PickleLocker_atconf_results/Data_big/7z/RF_seed101.csv"
"PickleLocker\_[tuner]\_results/Data\_[big\small]/[Data]/[model]\_[seed+num]\_[step+num].p"  <br>
- e.g. Model: "./PickleLocker_atconf_models/Data_big/7z/GP_seed101_step20.p"

## Supplementary
supplementary.pdf contains the specific best and worst models in this study.
