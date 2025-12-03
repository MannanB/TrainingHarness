
# python docker_scripts.py --build_push_base
# python docker_scripts.py --build_push_container
import subprocess
import os

from dotenv import load_dotenv

load_dotenv()

def build_and_push_base(username):
    base_image_tag = f"{username}/mltraining:latest"
    base_dockerfile_path = "./base"
    
    # Build the base image
    subprocess.run([
        "docker", "build", 
        "--platform", "linux/amd64", 
        "--tag", base_image_tag, 
        base_dockerfile_path
    ], check=True)
    
    # Push the base image to Docker Hub
    subprocess.run([
        "docker", "push", base_image_tag
    ], check=True)

def build_and_push_container(username):
    container_image_tag = f"{username}/runpodserverless:latest"
    container_dockerfile_path = "./container"
    
    # Build the container image
    subprocess.run([
        "docker", "build", 
        "--platform", "linux/amd64", 
        "--tag", container_image_tag, 
        container_dockerfile_path
    ], check=True)
    
    # Push the container image to Docker Hub
    subprocess.run([
        "docker", "push", container_image_tag
    ], check=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and push Docker images.")
    parser.add_argument("--build_push_base", action="store_true", help="Build and push the base image.")
    parser.add_argument("--build_push_container", action="store_true", help="Build and push the container image.")
    parser.add_argument("--username", type=str, help="Docker Hub username.")
    
    args = parser.parse_args()
    
    if args.build_push_base:
        build_and_push_base()
    
    if args.build_push_container:
        build_and_push_container()