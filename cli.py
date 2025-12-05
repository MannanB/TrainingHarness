import argparse
import subprocess
import sys
import os
import json
import runpod
import requests
import time

from dotenv import load_dotenv

load_dotenv()

DOCKER_USER = os.environ.get("DOCKER_USERNAME")
REPO_NAME_BASE = os.environ.get("DOCKER_BASE_REPO_NAME")
REPO_NAME_CONTAINER = os.environ.get("DOCKER_CONTAINER_REPO_NAME")

IMAGE_BASE = f"{DOCKER_USER}/{REPO_NAME_BASE}:latest"
IMAGE_CONTAINER = f"{DOCKER_USER}/{REPO_NAME_CONTAINER}:latest"

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def log(msg, type="info"):
    if type == "info": print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {msg}")
    elif type == "success": print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}")
    elif type == "error": print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}")
    elif type == "warn": print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {msg}")

def run_cmd(command):
    """Runs a shell command and streams output."""
    try:
        log(f"Running: {command}")
        # shell=True is used here for simplicity in a researcher CLI
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        log("Command failed.", "error")
        sys.exit(1)

def build_image(target):
    if target == "base":
        run_cmd(f"docker build --platform linux/amd64 -t {IMAGE_BASE} ./base")
    elif target in ["container", "runpod"]:
        # runpod is practically an alias for container for docker stuff specifically
        run_cmd(f"docker build --platform linux/amd64 -t {IMAGE_CONTAINER} ./container")

def push_image(target):
    if target == "base":
        run_cmd(f"docker push {IMAGE_BASE}")
    elif target in ["container", "runpod"]:
        run_cmd(f"docker push {IMAGE_CONTAINER}")

def update_runpod_endpoint():
    """
    Triggers a rolling update on Runpod using runpodctl.
    Requires runpodctl to be installed and configured.
    """
    log(f"Updating Endpoint {ENDPOINT_ID} to use image {IMAGE_CONTAINER}...")
    
    # Simple CLI command
    command = f"runpodctl.exe serverless endpoint update {ENDPOINT_ID} --image-name {IMAGE_CONTAINER}"
    
    run_cmd(command)
    log(f"Runpod update triggered successfully.", "success")

def handle_docker(args):
    target = args.target # base, container, runpod
    action = args.action # build, push, update, restart
    
    if target == "runpod" and action == "restart":
        update_runpod_endpoint()
        return

    # 1. Build Phase
    if action in ["build", "update"]:
        build_image(target)

    # 2. Push Phase
    if action in ["push", "update"]:
        push_image(target)

    # 3. Runpod Update Phase
    if action == "update" and target == "runpod":
        update_runpod_endpoint()


def cloud_run(exp_cfg):
    runpod.api_key = os.environ.get("RUNPOD_API_KEY")

    print(f"Submitting job to Runpod Endpoint {ENDPOINT_ID}...")

    # Submit the job. This is ASYNCHRONOUS.
    # It returns immediately with a job ID.
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    job_request = endpoint.run(exp_cfg)

    print(f"Job submitted!")
    print("Go to your W&B dashboard to monitor progress live.")
    print("Checking job status (will wait until complete)...")

    # Poll the job status until it's done
    while True:
        status = job_request.status()
        print(f"Job status: {status}")
        if status in ["COMPLETED", "FAILED"]:
            print("\n--- Job Finished ---")
            print(status)
            break
        time.sleep(10)

def local_run(exp_cfg):
    from container import rp_handler
    rp_handler.handler(exp_cfg)

def handle_jobs(args, extra_args):
    if args.action != "start":
        return
    
    with open(args.config, 'r') as f:
        exp_cfg = json.load(f)

    wandb_api_key = os.environ.get("W&B_API_KEY")
    wandb_user_name = os.environ.get("W&B_USER_NAME")
    wandb_project_name = os.environ.get("W&B_PROJECT_NAME")
    # autofill null entities
    if wandb_api_key and not exp_cfg["input"].get('W&B_API_KEY'):
        exp_cfg["input"]['W&B_API_KEY'] = wandb_api_key
    if wandb_project_name and not exp_cfg["input"].get('wandb_project_name'):
        exp_cfg["input"]['wandb_project_name'] = wandb_project_name
    if wandb_user_name and not exp_cfg["input"].get('wandb_user_name'):
        exp_cfg["input"]['wandb_user_name'] = wandb_user_name

    # print(f"Starting run {exp_cfg["input"].get('run_name', 'NO_RUN_NAME')} of project " +
    #       f"{exp_cfg["input"].get('project_name', 'NO_PROJECT_NAME')} (cloud={args.cloud})")
    log(f"Preparing job with config: {json.dumps(exp_cfg["input"], indent=2)}")

    if args.cloud:
        cloud_run(exp_cfg)
    else:
        local_run(exp_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Research Infrastructure CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Docker Command ---
    # Usage: python cli.py docker [build|push|update] [base|container|runpod]
    docker_parser = subparsers.add_parser("docker", help="Manage Docker Images")
    docker_parser.add_argument("action", choices=["build", "push", "update", "restart"], help="Action to perform")
    docker_parser.add_argument("target", choices=["base", "container", "runpod"], help="Target image")

    # --- Jobs Command ---
    # Usage: python cli.py jobs start --cloud --config ./inputs/cfg.json
    jobs_parser = subparsers.add_parser("jobs", help="Manage Jobs")
    jobs_parser.add_argument("action", choices=["start"], help="Job action")
    jobs_parser.add_argument("--cloud", action="store_true", help="Run the job on Runpod cloud.")
    jobs_parser.add_argument("--config", type=str, required=True, help="Path to the experiment config JSON file.")
    # Parse
    args, unknown = parser.parse_known_args()

    if args.command == "docker":
        handle_docker(args)
    elif args.command == "jobs":
        handle_jobs(args, unknown)
