
# Model-impact
This repository contains the data and code for the following paper: 
> Accuracy can Lie: On the Impact of Surrogate Model in Configuration Tuning

##  1. <a name='Tableofcontent'></a>Table of content
<!-- vscode-markdown-toc -->
1. [Table of content](#Tableofcontent)
2. [Introduction](#Introduction)
3. [Code and quick start](#Codeandquickstart)
4. [Datasets](#Datasets)
5. [Raw experiments results](#Rawexperimentsresults)
6. [RQ_supplementary](#RQ_supplementary)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  2. <a name='Introduction'></a>Introduction
To ease the expensive measurements during configuration tuning, it is natural to build a surrogate model as the replacement of the system and thereby the configuration performance can be cheaply evaluated. Yet, a stereotype therein is that the higher the model accuracy, the better the tuning result would be, or vice versa. This 'accuracy is all' belief drives our research community to build more and more accurate models and criticize a tuner due to the inaccuracy of the model it uses. However, this practice raises some previously unaddressed questions, e.g., whether the model and its accuracy are really that important for the tuning result? Do those somewhat small accuracy improvements reported (e.g., a few \% error reduction) in existing work really matter much to the tuners? What role does model accuracy play in the impact of tuning quality? To answer those related questions, in this paper, we conduct one of the largest-scale empirical studies to date---running over the period of 13 months $24\times7$---that covers 10 models, 17 tuners, and 29 systems from the existing works while under four different commonly used metrics, leading to 13,380 cases of investigation. Surprisingly, our key findings reveal that the accuracy can lie: there are a considerable number of cases where higher accuracy actually leads to no improvement in the tuning outcomes, or even worse, it can degrade the tuning quality. We also discover that the chosen models in most proposed tuners are sub-optimal and that the required \% of accuracy change to significantly improve tuning quality varies according to the range of model accuracy. From those, we provide in-depth discussions of the rationale behind and offer several lessons learned as well as insights for future opportunities. Most importantly, this work poses a clear message to the community that we should take one step back from the natural 'accuracy is all' belief for model-based configuration tuning.

##  3. <a name='Codeandquickstart'></a>Code and quick start
* code <br>
   -- Data (Datasets, the target need to start with "$<")<br>
   -- batch (Batch model-based tuners) <br>
   -- models (Surrogate models)<br>
   -- sequential (Sequential model-based tuners)<br>
   -- util (Util for tuners)<br>
   -- utils (Utils for models)<br>
   -- requirements.txt (Essential requirments need to be installed) <br>
   -- run (A simple run on system "7z", the working path is "./model-impact/code")

* Python 3.8+

To run the code, cd "./model-impact/code" as working path and install the essential requirements: 
```
pip install -r requirements.txt
```
And you can run below code to have a quick start:
```
python3 run.py
```


##  4. <a name='Datasets'></a>Datasets
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

Thanks for their efforts. Details of the datasets are given in our paper.

##  5. <a name='Rawexperimentsresults'></a>Raw experiments results

The experiment data reported in the work can be found at: https://zenodo.org/records/11172102. The naming rule follow as: <br>


*Result*: `PickleLocker_[tune]_results/[Data_big\small]/[Data]/[model]_[seed_num].csv`  <br>
- e.g. "./PickleLocker_atconf_results/Data_big/7z/RF_seed101.csv"<br>

<!-- *Model*: `PickleLocker\_[tuner]\_results/[Data_big\small]/[Data]/[model]\_[seed_num]\_[step_num].p`  <br>
- e.g. "./PickleLocker_atconf_models/Data_big/7z/GP_seed101_step20.p"<br> -->


##  6. <a name='RQ_supplementary'></a>RQ_supplementary
RQ_supplementary contains the supplementary files for our research questions.