install:
	pip install -r requirements.txt
run:
	python demo.py

tensorboard:
	tensorboard --logdir=ray_results/