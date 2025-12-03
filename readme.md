# Training Harness for ML/AI workloads

Supports local training and runpod.io serverless. Designed for ablation testing and small-scale training.
Uses wandb for logging for uploading models (HF Hub integration is planned)

## Setup:

1. Clone this repo
2. Create a .env file following the .env.example and populate it with required API keys (you can leave runpod stuff blank if local training). Make sure you make a wandb project
3. (optional for runpod) Run ```python docker_scripts.py --build_push_container --username "your docker username"``` (note this pulls from a premade base container. If you also want to build this, edit the Dockerfile in container to be from:your-container, and run the script with --build_push_base)
4. (optional for runpod) Use that docker image to deploy to runpod serverless
5. run ```python run_job.py --config ./inputs/example_input.json``` This will start a simple MNIST training for testing. Add --cloud to use runpod
6. View results in wandb

## Creating your own experiments

1. Create a new folder under ./container/projects with your project/experiment name
2. The entrypoint for the training code will be a main function within __init__.py in the root directory of your experiment. Look at the mnist for an example. It takes 2 args: the wandb run and the config dict passed from the input json in ./inputs
3. Code your training script
4. (optional for runpod) Rebuild your docker container and restart your serverless endpoint
5. Create an input json in ./inputs for your project with whatever config there is 
6. Run the job (for ablations, create multiple inputs and create multiple jobs from those inputs)
