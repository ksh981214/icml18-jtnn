# Bayesian Optimization, Use tqdm

For Bayesian optimization, we used the scripts from https://github.com/mkusner/grammarVAE

This requires you to install their customized Theano library. 
Please see https://github.com/mkusner/grammarVAE#bayesian-optimization for installation.

## Usage
First generate the latent representation of all training molecules:
```
python gen_latent.py --data ../data/zinc/train.txt --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --model ../molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4
```
This generates **latent_features.txt for latent vectors and other files for logP, synthetic accessability scores(targets.txt, logP_values.txt, SA_scores.txt, cycle_scores.txt)**

![gen_latent_result](./result_img/gen_latent_result.png)

- local 기준

To run Bayesian optimization:
```
SEED=1
mkdir results$SEED
python run_bo.py --vocab ../data/zinc/vocab.txt --model ../molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4 --save_dir results$SEED --hidden 450 --depth 3 --latent 56 --seed $SEED 

```
It performs five iterations of Bayesian optimization with EI heuristics, and saves discovered molecules in `results$SEED/` 
Following previous work, we tried `$SEED` from 1 to 10.

run_bo.py에서 latent representation m을 준다면 y(m)을 예측하는 a sparse Gaussian process(SGP)를 학습시킨다.

- y(m) = logP(m) - SA(m) - cycle(m)
    - SA(m)은 synthetic accessibility score 
    - cycle(m)은 6개 원자 이상을 가지는 ring의 count

최종적으로, valid_smiles.dat과 scores.dat을 저장한다.

여기서, theano라는 라이브러리를 깔아주어야하는데, 기존 latest버전으로는 해결할 수 없다.

따라서, [gVAE](https://github.com/mkusner/grammarVAE) 의 /Theano-master 로 들어가서 폴더를 /bo에 다운 받은 후

```
python setup.py install
```

을 실행해주어야한다.

![run_bo_result1](./result_img/run_bo_result1.png)
```
sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.001)
```

- iteration당 약 55분이 걸린다.

### Error

![run_bo_error](./error_img/run_bo_error1.png)
- /jtnn/jtnn_vae.py 264 lines 수정


To summarize results accross 10 runs:
```
python print_result.py
```

기존 코드는 10등분으로 나누어 10개의 result를 생성하지만, 깃허브의 코드는 result1에 대해서만 진행한다.

![print_result_img](./result_img/print_result_img.png)

- print_result.py 결과의 일부분
- 전체 결과는 [print_result.out](https://github.com/ksh981214/icml18-jtnn/blob/master/bo/print_result.out) 참조