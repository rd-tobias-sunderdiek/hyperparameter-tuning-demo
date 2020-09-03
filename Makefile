install:
	pip install -r requirements.txt
train:
	python train.py

tensorboard:
	tensorboard --logdir=ray_results/

random:
	python random_action.py

gif:
	python create_gif.py