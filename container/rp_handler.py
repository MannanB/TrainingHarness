import runpod
import time  
import os

import wandb

def handler(event):

    print(f"Worker Start")
    input = event['input']

    os.environ['WANDB_API_KEY'] = input['W&B_API_KEY']

    wandb.login()

    wandb_project_name = input.get('wandb_project_name', 'default_wandb_project')
    wandb_run_name = input.get('run_name', f'run_{int(time.time())}')
    wandb_user_name = input.get('wandb_user_name', None)

    project = input.get('project_name', 'default_project')
    cfg = input.get('config', {})

    run = wandb.init(
        project=wandb_project_name,
        entity=wandb_user_name,
        config=cfg,
        name=wandb_run_name,
    )

    try:
        project = __import__(f"projects.{project}", fromlist=['main'])
    except ModuleNotFoundError:
        # probably local run
        project = __import__(f"container.projects.{project}", fromlist=['main'])

    run_link = f"https://wandb.ai/{wandb_user_name}/{wandb_project_name}/runs/{run.id}"

    project.main(run, cfg)

    wandb.finish()

    return {"status": "completed", "run_id": run.id, "run_link": run_link}



# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })