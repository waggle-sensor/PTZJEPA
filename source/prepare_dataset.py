import logging
from pathlib import Path
import pickle
from typing import List, Union
import numpy as np
import time
import datetime
import subprocess
import os
import shutil
import pandas as pd
from PIL import Image
from torch import Tensor
import torch

from waggle.plugin import Plugin

logger = logging.getLogger(__name__)

try:
    # ! Note this assumes the code is running in a container
    coll_dir = Path("/collected_imgs")
    tmp_dir = Path("/imgs")
    persis_dir = Path("/persistence")
    coll_dir.mkdir(exist_ok=True, mode=0o777)
    tmp_dir.mkdir(exist_ok=True, mode=0o777)
    persis_dir.mkdir(exist_ok=True, mode=0o777)
except OSError:
    logger.warning(
        "Could not create directories, will use default paths and the code might break"
    )


def get_dirs():
    """
    Get the directories for persistent storage, collection, and temporary files.

    Returns:
        persis_dir (Path): The directory for persistent storage.
        coll_dir (Path): The directory for collection.
        tmp_dir (Path): The directory for temporary files.
    """
    return persis_dir, coll_dir, tmp_dir


# ---------------
# Prepare images
# ---------------
def grab_image(camera, args):
    if args.camerabrand == 0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand == 1:
        position = camera.get_ptz()

    pos_str = ",".join([str(p) for p in position])
    # ct stores current time
    ct = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    img_path = str(tmp_dir / f"{pos_str}_{ct}.jpg")
    try:
        camera.snap_shot(img_path)
    # TODO: need to check what kind of exception is raised
    except:
        with Plugin() as plugin:
            plugin.publish(
                "cannot.capture.image.from.camera", str(datetime.datetime.now())
            )
        return None
    return img_path


def tar_images(output_filename, folder_to_archive):
    try:
        cmd = ["tar", "cvf", output_filename, folder_to_archive]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        logger.info(output)
    except Exception:
        logger.exception("Error when tar images")


def publish_images():
    # run tar -cvf images.tar /imgs
    tar_images("images.tar", str(tmp_dir))
    # files = glob.glob("/imgs/*.jpg", recursive=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with Plugin() as plugin:
        ct = str(datetime.datetime.now())
        os.rename("images.tar", ct + "_images.tar")
        plugin.upload_file(ct + "_images.tar")


def verify_image(fp, try_fix=True):
    """
    Verifies the integrity of an image file.

    Args:
        fp (str): The file path of the image file to be verified.
        try_fix (bool): Whether to attempt to fix the image file if it is corrupted. Default is True.

    Returns:
        bool: True if the image file is valid or successfully fixed, False otherwise.
    """
    try:
        image = Image.open(fp)
        image.verify()
        if try_fix:
            # to fix corrupted images
            image = Image.open(fp)
            image.save(fp)
        return True
    except (OSError, IOError, SyntaxError) as e:
        logger.exception("Error: %s : %s", fp, e.strerror)
        return False


def collect_images(keepimages):
    coll_dir.mkdir(exist_ok=True, mode=0o777)
    # files = glob.glob("/imgs/*.jpg", recursive=True)
    for fp in tmp_dir.glob("*.jpg"):
        if verify_image(fp):
            try:
                shutil.copy(fp, coll_dir)
            except OSError as e:
                logger.error("Error: %s : %s", fp, e.strerror)
        if keepimages:
            dest = persis_dir / "collected_imgs"
            dest.mkdir(exist_ok=True, mode=0o777)
            # check mode of the directory
            if dest.stat().st_mode != 0o777:
                os.chmod(dest, 0o777)
            for fp in coll_dir.glob("*.jpg"):
                try:
                    dest_fp = shutil.copy(fp, dest)
                    os.chmod(dest_fp, 0o666)  # RW for all
                except OSError as e:
                    logger.error("Error: %s : %s", fp, e.strerror)


def get_images_from_storage(args):
    if not args.storedimages:
        operate_ptz(args)
    else:
        logger.info("Getting images from storage")
        if coll_dir.exists():
            # remove the directory and its contents
            # this ensure only images from persistence are used
            shutil.rmtree(coll_dir)
        coll_dir.mkdir(mode=0o777)
        img_dir = persis_dir / "collected_imgs"
        for fp in img_dir.glob("*.jpg"):
            try:
                shutil.copy(fp, coll_dir)
            except OSError as e:
                logger.error("Error: %s : %s", fp, e.strerror)


def prepare_images(label_dir="./"):
    """Prepare images for training.
    Check if all images are valid and remove invalid ones.
    """
    labels = []
    for fp in coll_dir.glob("*.jpg"):
        if verify_image(fp):
            labels.append(fp.stem)
        else:
            fp.unlink()
    df = pd.DataFrame(labels)
    label_path = Path(label_dir, "labels.txt")
    df.to_csv(label_path, header=False, index=False)
    os.chmod(label_path, 0o666)  # RW for all
    logger.info("Number of labels: %d", df.size)


def prepare_dreams():
    prepare_images()


# -----------------
# Prepare positions
# -----------------
def grab_position(camera, args):
    if args.camerabrand == 0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand == 1:
        position = camera.get_ptz()

    pos_str = ",".join([str(p) for p in position])

    return pos_str


def set_relative_position(camera, args, pan, tilt, zoom):
    print("pan ", pan)
    print("tilt ", tilt)
    print("zoom ", zoom)
    try:
        if args.camerabrand == 0:
            camera.relative_control(pan=pan, tilt=tilt, zoom=zoom)
        elif args.camerabrand == 1:
            camera.relative_move(rpan=pan, rtilt=tilt, rzoom=zoom)
    except:
        with Plugin() as plugin:
            plugin.publish(
                "cannot.set.camera.relative.position", str(datetime.datetime.now())
            )


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


def collect_positions(positions, current_time: None):
    directory = persis_dir / "collected_positions"
    directory.mkdir(exist_ok=True, mode=0o777, parents=True)

    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    with open(directory / f"positions_at_{current_time}.txt", "w") as fh:
        fh.write("\n".join(positions))
    change_ownership(directory)


def collect_commands(commands, current_time: None):
    directory = persis_dir / "collected_commands"
    directory.mkdir(exist_ok=True, mode=0o777, parents=True)

    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    with open(directory / f"commands_at_{current_time}.txt", "w") as fh:
        fh.write("\n".join(commands))
    change_ownership(directory)


def collect_embeds(embeds: List[Tensor], current_time: None):
    directory = persis_dir / "collected_embeds"
    directory.mkdir(exist_ok=True, mode=0o777, parents=True)

    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    fpath = directory / f"embeds_at_{current_time}.pt"
    torch.save(torch.vstack(embeds), fpath)
    change_ownership(directory)


def operate_ptz(args):
    if args.camerabrand == 0:
        logger.info("Importing Hanwha")
        from source import sunapi_control as camera_control
    elif args.camerabrand == 1:
        logger.info("Importing Axis")
        from source import vapix_control as camera_control

        # from source import onvif_control as camera_control
    else:
        raise ValueError("Not known camera brand number: ", args.camerabrand)

    iterations = args.iterations
    number_of_commands = args.movements

    try:
        Camera1 = camera_control.CameraControl(
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
    # reset the camera to its original position
    if args.camerabrand == 0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand == 1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    pan_modulation = 2
    tilt_modulation = 2
    zoom_modulation = 1 if args.camerabrand == 0 else 1000
    # if args.camerabrand==0:
    #     zoom_modulation = 1
    # elif args.camerabrand==1:
    #     zoom_modulation = 1000

    pan_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    pan_values *= pan_modulation
    tilt_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    tilt_values *= tilt_modulation
    zoom_values = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    zoom_values *= zoom_modulation

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


def change_ownership(folder):
    for subdir, dirs, files in os.walk(folder):
        os.chmod(subdir, 0o777)

        for File in files:
            os.chmod(os.path.join(subdir, File), 0o666)


def detect_plateau(loss_values, patience=5, threshold=1e-4):
    """
    Detects plateauing behavior in a loss curve.

    Parameters:
        loss_values (list or numpy array): List or array containing the loss values over epochs.
        patience (int): Number of epochs with no improvement to wait before stopping.
        threshold (float): Threshold for the change in loss to be considered as plateauing.

    Returns:
        plateaued (bool): True if the loss has plateaued, False otherwise.
    """
    if len(loss_values) < patience + 1:
        return False  # Not enough data to detect plateauing

    recent_losses = loss_values[-patience:]
    mean_loss = np.mean(recent_losses)
    current_loss = loss_values[-1]

    if np.abs(current_loss - mean_loss) < threshold:
        return True  # Loss has plateaued
    else:
        return False  # Loss has not plateaued
