import runpod
import time  
import os

import wandb

def handler(event):

    print(f"Worker Start")
    input = event['input']

    use_wandb = input.get('wandb_logging', True)
    project = input.get('project_name', 'default_project')
    cfg = input.get('config', {})

    run = None
    run_link = None
    run_id = None
    if use_wandb:
        os.environ['WANDB_API_KEY'] = input['W&B_API_KEY']
        wandb.login()

        wandb_project_name = input.get('wandb_project_name', 'default_wandb_project')
        wandb_run_name = input.get('run_name', f'run_{int(time.time())}')
        wandb_user_name = input.get('wandb_user_name', None)

        run = wandb.init(
            project=wandb_project_name,
            entity=wandb_user_name,
            config=cfg,
            name=wandb_run_name,
        )
        run_link = f"https://wandb.ai/{wandb_user_name}/{wandb_project_name}/runs/{run.id}"
        run_id = run.id

    try:
        project = __import__(f"projects.{project}", fromlist=['main'])
    except ModuleNotFoundError:
        # probably local run
        project = __import__(f"container.projects.{project}", fromlist=['main'])

    project.main(run, cfg)

    if use_wandb:
        wandb.finish()

    return {"status": "completed", "run_id": run_id, "run_link": run_link}



# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })