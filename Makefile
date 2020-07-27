install:
	pip install -r requirements.txt
train:
	python train.py

tensorboard:
	tensorboard --logdir=ray_results/

tensorboard_demo:
	tensorboard --logdir=ray_results_demo/

gif:
	python create_gif.py