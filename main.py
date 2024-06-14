# import sys
# sys.path.append("/app/source")

import time
import datetime
import os
import shutil
import glob
import yaml
import csv
import traceback
import subprocess
from pathlib import Path

import argparse

import numpy as np
import pandas as pd

from PIL import Image

from waggle.plugin import Plugin

from source.run_jepa import run as run_jepa
from source.run_rl import run as run_rl
from source.env_interaction import run as env_inter

import torch


coll_dir = Path("/collected_imgs")
coll_dir.mkdir(exist_ok=True, mode=0o777)
tmp_dir = Path("/imgs")
tmp_dir.mkdir(exist_ok=True, mode=0o777)
persis_dir = Path("/persistence")
persis_dir.mkdir(exist_ok=True, mode=0o777)


def set_random_position(camera, args):
    if args.camerabrand == 0:
        pan_pos = np.random.randint(0, 360)
        tilt_pos = np.random.randint(-20, 90)
        zoom_pos = np.random.randint(1, 2)
    elif args.camerabrand == 1:
        pan_pos = np.random.randint(-180, 180)
        tilt_pos = np.random.randint(-180, 180)
        zoom_pos = np.random.randint(100, 200)
    try:
        if args.camerabrand == 0:
            camera.absolute_control(float(pan_pos), float(tilt_pos), float(zoom_pos))
        elif args.camerabrand == 1:
            camera.absolute_move(float(pan_pos), float(tilt_pos), int(zoom_pos))
    except:
        with Plugin() as plugin:
            plugin.publish(
                "cannot.set.camera.random.position", str(datetime.datetime.now())
            )

    time.sleep(1)


def grab_image(camera, args):
    if args.camerabrand == 0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand == 1:
        position = camera.get_ptz()

    pos_str = str(position[0]) + "," + str(position[1]) + "," + str(position[2]) + " "
    # ct stores current time
    ct = str(datetime.datetime.now())
    try:
        camera.snap_shot(str(tmp_dir / (pos_str + ct + ".jpg")))
    except:
        with Plugin() as plugin:
            plugin.publish(
                "cannot.capture.image.from.camera", str(datetime.datetime.now())
            )


def tar_images(output_filename, folder_to_archive):
    try:
        cmd = ["tar", "cvf", output_filename, folder_to_archive]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        print(output)
    except Exception:
        print(f"E: {traceback.format_exc()}")


def publish_images():
    # run tar -cvf images.tar /imgs
    tar_images("images.tar", str(tmp_dir))
    # files = glob.glob("/imgs/*.jpg", recursive=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with Plugin() as plugin:
        ct = str(datetime.datetime.now())
        os.rename("images.tar", ct + "_images.tar")
        plugin.upload_file(ct + "_images.tar")


def collect_images(keepimages):
    coll_dir.mkdir(exist_ok=True, mode=0o777)
    # files = glob.glob("/imgs/*.jpg", recursive=True)
    for fp in tmp_dir.glob("*.jpg"):
        try:
            fp.rename(coll_dir / fp.name)
        except OSError as e:
            print("Error: %s : %s" % (fp, e.strerror))

    if keepimages:
        dest = persis_dir / "collected_imgs"
        dest.mkdir(exist_ok=True, mode=0o777)
        for fp in coll_dir.glob("*.jpg"):
            try:
                shutil.copy(fp, dest)
            except OSError as e:
                print("Error: %s : %s" % (fp, e.strerror))


def prepare_images():
    """Prepare images for training.
    Check if all images are valid and remove invalid ones.
    """
    labels = []
    for fp in coll_dir.glob("*.jpg"):
        try:
            image = Image.open(fp).load()
            image.verify()
            image.save(fp) # to fix corrupted images
            labels.append(fp.stem)
        except (OSError, IOError, SyntaxError) as e:
            fp.unlink()
            print("Error: %s : %s" % (fp, e.strerror))
    df = pd.DataFrame(labels)
    df.to_csv("./labels.txt", header=None, index=False)
    print("Number of labels: ", df.size)


def prepare_dreams():
    # ! Might need to change this part in future
    # ! this function duplicates the function prepare_images
    labels = []
    files = glob.glob("/collected_imgs/*.jpg", recursive=True)
    for f in files:
        try:
            image = Image.open(f)
            image.save(f)
            labels.append(os.path.splitext(os.path.basename(f))[0])
        except OSError as e:
            os.remove(f)
            print("Error: %s : %s" % (f, e.strerror))

    df = pd.DataFrame(labels)
    df.to_csv("./labels", header=None, index=False)
    print("Number of labels: ", df.size)


def operate_ptz(args):
    if args.camerabrand == 0:
        print("Importing Hanwha")
        from source import sunapi_control as sunapi_control
    elif args.camerabrand == 1:
        print("Importing Axis")
        from source import vapix_control as sunapi_control

        # from source import onvif_control as sunapi_control
    else:
        raise ValueError("Not known camera brand number: ", args.camerabrand)

    iterations = args.iterations
    number_of_commands = args.movements

    try:
        Camera1 = sunapi_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except:
        with Plugin() as plugin:
            plugin.publish(
                "cannot.get.camera.from.ip",
                args.cameraip,
                timestamp=datetime.datetime.now(),
            )
            plugin.publish(
                "cannot.get.camera.from.un",
                args.username,
                timestamp=datetime.datetime.now(),
            )
            plugin.publish(
                "cannot.get.camera.from.pw",
                args.password,
                timestamp=datetime.datetime.now(),
            )

    if args.camerabrand == 0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand == 1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    pan_modulation = 2
    tilt_modulation = 2
    zoom_modulation = 1

    pan_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    pan_values = pan_values * pan_modulation
    tilt_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    tilt_values = tilt_values * tilt_modulation
    if args.camerabrand == 0:
        zoom_values = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    elif args.camerabrand == 1:
        zoom_values = 100 * np.array([-2, -1, 0, 1, 2])

    zoom_values = zoom_values * zoom_modulation

    with Plugin() as plugin:
        plugin.publish(
            "starting.new.image.collection.the.number.of.iterations.is", iterations
        )
        plugin.publish(
            "the.number.of.images.recorded.by.iteration.is", number_of_commands
        )

    if coll_dir.exists():
        shutil.rmtree(coll_dir, ignore_errors=True)
    for iteration in range(iterations):
        with Plugin() as plugin:
            plugin.publish("iteration.number", iteration)

        tmp_dir.mkdir(exist_ok=True, mode=0o777)
        PAN = np.random.choice(pan_values, number_of_commands)
        TILT = np.random.choice(tilt_values, number_of_commands)
        ZOOM = np.random.choice(zoom_values, number_of_commands)
        set_random_position(camera=Camera1, args=args)
        grab_image(camera=Camera1, args=args)

        for pan, tilt, zoom in zip(PAN, TILT, ZOOM):
            try:
                if args.camerabrand == 0:
                    Camera1.relative_control(pan=pan, tilt=tilt, zoom=zoom)
                elif args.camerabrand == 1:
                    Camera1.relative_move(rpan=pan, rtilt=tilt, rzoom=zoom)
            except:
                with Plugin() as plugin:
                    plugin.publish(
                        "cannot.set.camera.relative.position",
                        str(datetime.datetime.now()),
                    )

            grab_image(camera=Camera1, args=args)

        # publish_images()
        collect_images(args.keepimages)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.camerabrand == 0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand == 1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    with Plugin() as plugin:
        plugin.publish("finishing.image.collection", str(datetime.datetime.now()))


def get_images_from_storage(args):
    if not args.storedimages:
        operate_ptz(args)
    else:
        if coll_dir.exists():
            # remove the directory and its contents
            # this ensure only images from persistence are used
            shutil.rmtree(coll_dir)
        coll_dir.mkdir(mode=0o777)
        for fp in persis_dir.glob("*.jpg"):
            try:
                shutil.copy(fp, coll_dir)
            except OSError as e:
                print("Error: %s : %s" % (fp, e.strerror))


def pretraining_wrapper(arguments):
    training_complete = False
    while not training_complete:
        get_images_from_storage(arguments)
        # operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, "train")


def pretraining_world_model_wrapper(arguments):
    training_complete = False
    while not training_complete:
        get_images_from_storage(arguments)
        # operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, "world_model")


def dreamer_wrapper(arguments):
    operate_ptz(arguments)
    prepare_images()
    number_of_iterations = 10
    for itr in range(number_of_iterations):
        run_jepa(arguments.fname, "dreamer")


def behavior_learning(arguments):
    training_complete = False
    while not training_complete:
        prepare_dreams()
        training_complete = run_rl(arguments.fname, "train_agent")


def environment_interaction(arguments):
    interaction_complete = env_inter(arguments, arguments.fname, "navigate_env")
    print("interaction_complete: ", interaction_complete)


def lifelong_learning(arguments):
    while True:
        operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, "world_model")
        training_complete = run_jepa(arguments.fname, "dreamer")
        training_complete = run_rl(arguments.fname, "train_agent")


def main():
    parser = argparse.ArgumentParser("PTZ JEPA")

    # PTZ sampler
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-si",
        "--storedimages",
        action="store_true",
        help="Gather images from determined storage location",
    )
    parser.add_argument(
        "-cb",
        "--camerabrand",
        help="An integer for each accepted camera brand (default=0). 0 is Hanwha, 1 is Axis.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-it",
        "--iterations",
        help="An integer with the number of iterations (PTZ rounds) to be run (default=10).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-mv",
        "--movements",
        help="An integer with the number of movements in each PTZ round to be run (default=10).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-un",
        "--username",
        help="The username of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-pw",
        "--password",
        help="The password of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ip", "--cameraip", help="The ip of the PTZ camera.", type=str, default=""
    )
    parser.add_argument(
        "-rm",
        "--run_mode",
        help="The mode to run the code.",
        choices=[
            "train",
            "world_model_train",
            "dream",
            "agent_train",
            "env_interaction",
            "lifelong",
        ],
        type=str,
        default="train",
    )

    # Joint Embedding Predictive Architecture (JEPA)
    parser.add_argument(
        "-fn",
        "--fname",
        type=str,
        help="name of config file to load",
        default="./configs/Config_file.yaml",
    )
    # default='/percistence/configs/in1k_vith14_ep300.yaml')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if args.run_mode == "train":
        pretraining_wrapper(args)
    elif args.run_mode == "world_model_train":
        pretraining_world_model_wrapper(args)
    elif args.run_mode == "dream":
        dreamer_wrapper(args)
    elif args.run_mode == "agent_train":
        behavior_learning(args)
    elif args.run_mode == "env_interaction":
        environment_interaction(args)
    elif args.run_mode == "lifelong":
        lifelong_learning(args)

    print("DONE!")


if __name__ == "__main__":
    main()
