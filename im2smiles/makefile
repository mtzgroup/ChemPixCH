# Run `$ make train-<run-name>` to train NN and `$ make eval-<run-name>` to evaluate results

# Check that environment is working with small dataset
train-small:
	python train.py --data=configs/data_small.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/small/
eval-small:
	python evaluate_txt.py --results=results/small/


#### Figure 2: clean RDKit images ####

train-clean-RDKit-10K:
	python train.py --data=configs/data_clean-RDKit-10K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/clean-RDKit-10K/
eval-clean-RDKit-10K:
	python evaluate_txt.py --results=results/clean-RDKit-10K/

train-clean-RDKit-50K:
	python train.py --data=configs/data_clean-RDKit-50K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/clean-RDKit-50K/
eval-clean-RDKit-50K:
	python evaluate_txt.py --results=results/clean-RDKit-50K/

train-clean-RDKit-100K:
	python train.py --data=configs/data_clean-RDKit-100K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/clean-RDKit-100K/
eval-clean-RDKit-100K:
	python evaluate_txt.py --results=results/clean-RDKit-100K/

train-clean-RDKit-200K:
	python train.py --data=configs/data_clean-RDKit-200K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/clean-RDKit-200K/
eval-clean-RDKit-200K:
	python evaluate_txt.py --results=results/clean-RDKit-200K/

train-clean-RDKit-500K:
	python train.py --data=configs/data_clean-RDKit-500K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/clean-RDKit-500K/
eval-clean-RDKit-500K:
	python evaluate_txt.py --results=results/clean-RDKit-500K/



#### Figure 7a - Stages of synthetic data generation pipeline ####
train-SD-stage-rdkitp:
	python train.py --data=configs/data_SD-stage-rdkitp.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-stage-rdkitp/
eval-SD-stage-rdkitp:
	python evaluate_txt.py --results=results/SD-stage-rdkitp/

train-SD-stage-rdkitp-aug:
	python train.py --data=configs/data_SD-stage-rdkitp-aug.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-stage-rdkitp-aug/
eval-SD-stage-rdkitp-aug:
	python evaluate_txt.py --results=results/SD-stage-rdkitp-aug/

train-SD-stage-rdkitp-aug-bkg:
	python train.py --data=configs/data_SD-stage-rdkitp-aug-bkg.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-stage-rdkitp-aug-bkg/
eval-SD-stage-rdkitp-aug-bkg:
	python evaluate_txt.py --results=results/SD-stage-rdkitp-aug-bkg/

train-SD-stage-rdkitp-aug-bkg-deg:
	python train.py --data=configs/data_SD-stage-rdkitp-aug-bkg-deg.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-stage-rdkitp-aug-bkg-deg/
eval-SD-stage-rdkitp-aug-bkg-deg:
	python evaluate_txt.py --results=results/SD-stage-rdkitp-aug-bkg-deg/



#### Figure 7b - Size tests for synthetic datasets ####
train-SD-sizes-50K:
	python train.py --data=configs/data_SD-sizes-50K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-sizes-50K/
eval-SD-sizes-50K:
	python evaluate_txt.py --results=results/SD-sizes-50K/

train-SD-sizes-100K:
	python train.py --data=configs/data_SD-sizes-100K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-sizes-100K/
eval-SD-sizes-100K:
	python evaluate_txt.py --results=results/SD-sizes-100K/

train-SD-sizes-200K:
	python train.py --data=configs/data_SD-sizes-200K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-sizes-200K/
eval-SD-sizes-200K:
	python evaluate_txt.py --results=results/SD-sizes-200K/

train-SD-sizes-500K:
	python train.py --data=configs/data_SD-sizes-500K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/SD-sizes-500K/
eval-SD-sizes-500K:
	python evaluate_txt.py --results=results/SD-sizes-500K/



#### Figure 8a - Size tests for synthetic training sets with hand-drawn validation set ####

train-HDval-50K:
	python train.py --data=configs/data_HDval-50K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDval-50K/
eval-HDval-50K:
	python evaluate_txt.py --results=results/HDval-50K/

train-HDval-100K:
	python train.py --data=configs/data_HDval-100K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDval-100K/
eval-HDval-100K:
	python evaluate_txt.py --results=results/HDval-100K/

train-HDval-200K:
	python train.py --data=configs/data_HDval-200K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDval-200K/
eval-HDval-200K:
	python evaluate_txt.py --results=results/HDval-200K/

train-HDval-500K:
	python train.py --data=configs/data_HDval-500K.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDval-500K/
eval-HDval-500K:
	python evaluate_txt.py --results=results/HDval-500K/



#### Figure 8b - Ratio of synthetic:hand-drawn training data ####

train-HDtrain-100_0:
	python train.py --data=configs/data_HDtrain-100_0.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDtrain-100_0/
eval-HDtrain-100_0:
	python evaluate_txt.py --results=results/HDtrain-100_0/

train-HDtrain-90_10:
	python train.py --data=configs/data_HDtrain-90_10.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDtrain-90_10/
eval-HDtrain-90_10:
	python evaluate_txt.py --results=results/HDtrain-90_10/

train-HDtrain-50_50:
	python train.py --data=configs/data_HDtrain-50_50.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDtrain-50_50/
eval-HDtrain-50_50:
	python evaluate_txt.py --results=results/HDtrain-50_50/

train-HDtrain-10_90:
	python train.py --data=configs/data_HDtrain-10_90.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDtrain-10_90/
eval-HDtrain-10_90:
	python evaluate_txt.py --results=results/HDtrain-10_90/

train-HDtrain-0_100:
	python train.py --data=configs/data_HDtrain-0_100.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/HDtrain-0_100/
eval-HDtrain-0_100:
	python evaluate_txt.py --results=results/HDtrain-0_100/

