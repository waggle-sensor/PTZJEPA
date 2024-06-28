# Pan Tilt Zoom Joint Embedding Predictive Architecture (PTZJEPA)

This app uses the JEPA framework to process PTZ images

## Building the container

The following command will build the container for both x86 and ARM64 architectures and push the image to the DockerHub registry.
To run it, you need to have Docker installed and buildx enabled.
[See here for more information on how to get buildx](https://docs.docker.com/build/architecture/#buildx).

```bash
docker buildx build --platform=linux/amd64,linux/arm64 -t some_tag_name -f Dockerfile --push .
```

## Running the container

The following command will run the container on the host machine.

```bash
sudo docker run --name train-model --rm --gpus all \
-v /your_model_directory:/persistence -v /parent_path/PTZJEPA:/app \
your_docker_image_tag python /app/main.py main_function_commands 2>&1 |
tee /path_to_log_file/log.out
```

The main function arguments are the same as the ones in the main.py file.
I have already build an image to run the task, so you can substitute the image name in the command above with `brookluo/ptzjepa:py310-cu116` to run the task on a Dell blade.
