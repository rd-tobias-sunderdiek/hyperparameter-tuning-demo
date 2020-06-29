install:
	pip install -r requirements.txt
train:
	python train.py

tensorboard:
	tensorboard --logdir=ray_results/

gif:
	python create_gif.py