# Molecule Generation
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/icml18-jtnn
```
Our ZINC dataset is in `icml18-jtnn/data/zinc` (copied from https://github.com/mkusner/grammarVAE). 
We follow the same train/dev/test split as previous work. 

## Deriving Vocabulary, 새로운 데이터셋을 쓸 경우 vocab을 제공 
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python ../jtnn/mol_tree.py < ../data/zinc/all.txt
```
This gives you the vocabulary of cluster labels over the dataset `all.txt`.

![deriving vocab resulg img](./result_img/deriving_vocabulary_result.png)

## Training
We trained VAE model in two phases:
1. We train our model for three epochs **without KL regularization term** (So we are essentially training an autoencoder).
Pretrain our model as follows (with hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
mkdir pre_model/
nohup python pretrain.py --train ../data/zinc/train.txt --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --save_dir pre_model/ > ./pre_model/LOG.out &
```
PyTorch by default uses all GPUs, setting flag `CUDA_VISIBLE_DEVICES=0` forces PyTorch to use the first GPU (1 for second GPU and so on).

The final model is saved at pre_model/model.2

### Error in Pre_training
![error in pretrain_1](./error_img/pretrain_err_1.png)
- jtnn_vae.py 166 lines 수정

2. Train out model **with KL regularization**, with constant regularization weight $beta$. 
We found setting beta > 0.01 greatly damages reconstruction accuracy.
```
mkdir new_models/
nohup python vaetrain.py --train ../data/zinc/train.txt --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0007 --beta 0.005 --model pre_model/model.2 --save_dir new_models/ > ./new_models/LOG.out &
```

## Testing
train된 모델로부터 새로운 분자들을 뽑아보고 싶을 때
```
python sample.py --nsample 100 --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --model MPNVAE-h450-L56-d3-beta0.005/model.iter-4
```
This script prints each line the SMILES string of each molecule. `prior_mols.txt` contains these SMILES strings.

![test1_result](./result_img/test1_result.png)

For molecule reconstruction, run  
```
python reconstruct.py --test ../data/zinc/test.txt --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --model MPNVAE-h450-L56-d3-beta0.005/model.iter-4
```
hyper parameter tuning을 위해 test.txt 대신 valid.txt를 사용해서 실행이 가능하다.

![test2_result](./result_img/test2_result.png)

이런 식으로 계속 나온다. 어느 정도 0.76에 수렴. 수치는 reconstruction의 정확도를 말한다.

## MOSES Benchmark Results
We also trained our model over MOSES benchmark dataset. The trained model is saved in `moses-h450L56d3beta0.5/`. To generate samples from our model, run
```
python sample.py --nsample 30000 --vocab ../data/moses/vocab.txt \
--hidden 450 --depth 3 --latent 56 --stereo 0 \
--model moses-h450L56d3beta0.5/model.iter-2
```
where `--stereo 0` means the model will not infer stereochemistry (because molecules in MOSES dataset does not contain stereochemistry). This should give you the same samples in [moses-h450L56d3beta0.5/samples.txt](moses-h450L56d3beta0.5/samples.txt). The result is as follows:
```
valid = 0.9991
unique@1000 = 1.0
unique@10000 = 0.9997
FCD/Test = 0.9770302413177916
SNN/Test = 0.522326049871644
Frag/Test = 0.9950979926332992
Scaf/Test = 0.8655089872053796
FCD/TestSF = 1.5980327517965094
SNN/TestSF = 0.4996388119246172
Frag/TestSF = 0.9926974330760409
Scaf/TestSF = 0.1174452677242035
IntDiv = 0.8562054073435843
IntDiv2 = 0.8503170074513857
Filters = 0.9743769392453208
logP = 0.02464815889709815
SA = 0.15781023266502325
QED = 2.1869624593648385e-05
NP = 0.0962078166269753
weight = 8.657725423864576
```
