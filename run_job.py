import runpod
import os
import time

from dotenv import load_dotenv

load_dotenv()

def cloud_run(exp_cfg):
    runpod.api_key = os.environ.get("RUNPOD_API_KEY")
    ENDPOINT_ID = os.environ.get("ENDPOINT_ID")


    print(f"Submitting job to Runpod Endpoint {ENDPOINT_ID}...")

    # Submit the job. This is ASYNCHRONOUS.
    # It returns immediately with a job ID.
    job_request = runpod.api.run_async(ENDPOINT_ID, exp_cfg)
    job_id = job_request['id']

    print(f"Job submitted! Job ID: {job_id}")
    print("Go to your W&B dashboard to monitor progress live.")
    print("Checking job status (will wait until complete)...")

    # Poll the job status until it's done
    while True:
        status = runpod.api.job_status(job_id)
        print(f"Job status: {status['status']}")
        if status['status'] in ["COMPLETED", "FAILED"]:
            print("\n--- Job Finished ---")
            print(status)
            break
        time.sleep(10)

def local_run(exp_cfg):
    from container import rp_handler
    rp_handler.handler(exp_cfg)

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run training job locally or on Runpod.")
    parser.add_argument("--cloud", action="store_true", help="Run the job on Runpod cloud.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config JSON file.")

    args = parser.parse_args()

    # Load experiment configuration from JSON file
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

    print(f"Starting run {exp_cfg["input"].get('run_name', 'NO_RUN_NAME')} of project " +
          f"{exp_cfg["input"].get('project_name', 'NO_PROJECT_NAME')} (cloud={args.cloud})")
    input("Press Enter to continue...")
    if args.cloud:
        cloud_run(exp_cfg)
    else:
        local_run(exp_cfg)