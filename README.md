# IFA-DTI
# Environment
python 3.7
pytorch 1.7.1
rdkit
pytorch geometric
Cuda 11.7
pytorch-optimizer: pip install -U pytorch-optimizer
# How to run

预处理数据集：运行 ./code/data_prepare.py (celegans和human数据集) 或者data_prepare_bDB.py(Binding_DB数据集)。

模型训练：
celegans和human数据集：运行./code/train.py。 在train.py中修改以下相应代码：

train('human', 0, 0.98, 'inter', False, 1, True)

train('celegans', 0, 0.98, 'inter', False, 1, True)

Binding_DB数据集：运行./code/train_bDB.py。