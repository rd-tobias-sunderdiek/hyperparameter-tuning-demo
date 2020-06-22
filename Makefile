install:
	pip install -r requirements
run:
	python demo.py

tensorboard:
	tensorboard --logdir=ray_results/