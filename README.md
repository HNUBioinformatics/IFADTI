# IFA-DTI
# Environment
python 3.7
pytorch 1.7.1
rdkit
pytorch geometric
Cuda 11.7
pytorch-optimizer: pip install -U pytorch-optimizer
# How to run

Ԥ�������ݼ������� ./code/data_prepare.py (celegans��human���ݼ�) ����data_prepare_bDB.py(Binding_DB���ݼ�)��

ģ��ѵ����
celegans��human���ݼ�������./code/train.py�� ��train.py���޸�������Ӧ���룺

train('human', 0, 0.98, 'inter', False, 1, True)

train('celegans', 0, 0.98, 'inter', False, 1, True)

Binding_DB���ݼ�������./code/train_bDB.py��