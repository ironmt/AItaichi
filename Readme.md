## 说明
本项目依赖关系复杂，只能通过yaml安装，在conda隔离环境内运行，请按照下面的方法进行安装

## Installation
```shell
cd /home/service/competition/17501024327/pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Run
```shell
conda activate pyskl
python /home/service/competition/17501024327/PoseTracking.py
```

## 功能
把对应文件夹（位置见PoseTracking.py）下的八段锦/五禽戏视频进行分类和打分