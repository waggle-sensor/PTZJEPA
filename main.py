# import sys
# sys.path.append("/app/source")

import time
import datetime
import os
import glob
import yaml
import csv
import os.path
import traceback
import subprocess

import argparse

import numpy as np
import pandas as pd

from PIL import Image

from waggle.plugin import Plugin

from source.run_jepa import run as run_jepa
from source.run_rl import run as run_rl



def set_random_position(camera, args):
    if args.camerabrand==0:
        pan_pos = np.random.randint(0, 360)
        tilt_pos = np.random.randint(-20, 90)
        zoom_pos = np.random.randint(1, 2)
    elif args.camerabrand==1:
        pan_pos = np.random.randint(-180, 180)
        tilt_pos = np.random.randint(-180, 180)
        zoom_pos = np.random.randint(100, 200)
    try:
        if args.camerabrand==0:
            camera.absolute_control(float(pan_pos), float(tilt_pos), float(zoom_pos))
        elif args.camerabrand==1:
            camera.absolute_move(float(pan_pos), float(tilt_pos), int(zoom_pos))
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.set.camera.random.position', str(datetime.datetime.now()))

    time.sleep(1)

def grab_image(camera, args):
    if args.camerabrand==0:
        position = camera.requesting_cameras_position_information()
    elif args.camerabrand==1:
        position = camera.get_ptz()

    pos_str = str(position[0]) + ',' + str(position[1]) + ',' + str(position[2]) + ' '
    # ct stores current time
    ct = str(datetime.datetime.now())
    try:
        camera.snap_shot('./imgs/' + pos_str + ct + '.jpg')
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.capture.image.from.camera', str(datetime.datetime.now()))

def tar_images(output_filename, folder_to_archive):
    try:
        cmd = ['tar', 'cvf', output_filename, folder_to_archive]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        print(output)
    except Exception:
        print(f"E: {traceback.format_exc()}")

def publish_images():
    # run tar -cvf images.tar ./imgs
    tar_images('images.tar', './imgs')
    files = glob.glob('./imgs/*.jpg', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    with Plugin() as plugin:
        ct = str(datetime.datetime.now())
        os.rename('images.tar', ct + '_images.tar')
        plugin.upload_file(ct + '_images.tar')


def collect_images():
    directory = './collected_imgs'
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = glob.glob('./imgs/*.jpg', recursive=True)
    for f in files:
        try:
            os.rename(f, os.path.join(directory, os.path.basename(f)))
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def prepare_images():
    labels = []
    files = glob.glob('./collected_imgs/*.jpg', recursive=True)
    for f in files:
        try:
            image = Image.open(f)
            image.save(f)
            labels.append(os.path.splitext(os.path.basename(f))[0])
        except OSError as e:
            os.remove(f)
            print("Error: %s : %s" % (f, e.strerror))

    df = pd.DataFrame(labels)
    df.to_csv('./labels', header=None, index=False)
    print('Number of labels: ', df.size)





def prepare_dreams():
    labels = []
    files = glob.glob('./collected_imgs/*.jpg', recursive=True)
    for f in files:
        try:
            image = Image.open(f)
            image.save(f)
            labels.append(os.path.splitext(os.path.basename(f))[0])
        except OSError as e:
            os.remove(f)
            print("Error: %s : %s" % (f, e.strerror))

    df = pd.DataFrame(labels)
    df.to_csv('./labels', header=None, index=False)
    print('Number of labels: ', df.size)





def operate_ptz(args):
    if args.camerabrand==0:
        print('Importing Hanwha')
        from source import sunapi_control as sunapi_control
    elif args.camerabrand==1:
        print('Importing Axis')
        from source import vapix_control as sunapi_control
        #from source import onvif_control as sunapi_control
    else:
        print('Not known camera brand number: ', args.camerabrand)

    iterations = args.iterations
    number_of_commands = args.movements

    try:
        Camera1 = sunapi_control.CameraControl(args.cameraip, args.username, args.password)
    except:
        with Plugin() as plugin:
            plugin.publish('cannot.get.camera.from.ip', args.cameraip, timestamp=datetime.datetime.now())
            plugin.publish('cannot.get.camera.from.un', args.username, timestamp=datetime.datetime.now())
            plugin.publish('cannot.get.camera.from.pw', args.password, timestamp=datetime.datetime.now())
            

    if args.camerabrand==0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand==1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    pan_modulation = 2
    tilt_modulation = 2
    zoom_modulation = 1

    pan_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    pan_values = pan_values * pan_modulation
    tilt_values = np.array([-5, -1, -0.1, 0, 0.1, 1, 5])
    tilt_values = tilt_values * tilt_modulation
    if args.camerabrand==0:
        zoom_values = np.array([-0.2, -0.1, 0, 0.1, 0.2])
    elif args.camerabrand==1:
        zoom_values = 100*np.array([-2, -1, 0, 1, 2])

    zoom_values = zoom_values * zoom_modulation

    with Plugin() as plugin:
        plugin.publish('starting.new.image.collection.the.number.of.iterations.is', iterations)
        plugin.publish('the.number.of.images.recorded.by.iteration.is', number_of_commands)

    for iteration in range(iterations):
        with Plugin() as plugin:
            plugin.publish('iteration.number', iteration)

        os.mkdir('./imgs')
        PAN = np.random.choice(pan_values, number_of_commands)
        TILT = np.random.choice(tilt_values, number_of_commands)
        ZOOM = np.random.choice(zoom_values, number_of_commands)
        set_random_position(camera=Camera1, args=args)
        grab_image(camera=Camera1, args=args)

        for (pan, tilt, zoom) in zip(PAN, TILT, ZOOM):
            try:
                if args.camerabrand==0:
                    Camera1.relative_control(pan=pan, tilt=tilt, zoom=zoom)
                elif args.camerabrand==1:
                    Camera1.relative_move(rpan=pan, rtilt=tilt, rzoom=zoom)
            except:
                with Plugin() as plugin:
                    plugin.publish('cannot.set.camera.relative.position', str(datetime.datetime.now()))

            grab_image(camera=Camera1, args=args)

        #publish_images()
        collect_images()
        os.rmdir('./imgs')


    if args.camerabrand==0:
        Camera1.absolute_control(1, 1, 1)
        time.sleep(1)
    elif args.camerabrand==1:
        Camera1.absolute_move(1, 1, 1)
        time.sleep(1)

    with Plugin() as plugin:
        plugin.publish('finishing.image.collection', str(datetime.datetime.now()))


def pretraining_wrapper(arguments):
    training_complete = False
    while not training_complete:
        operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, 'train')

def pretraining_world_model_wrapper(arguments):
    training_complete = False
    while not training_complete:
        operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, 'world_model')

def dreamer_wrapper(arguments):
    operate_ptz(arguments)
    prepare_images()
    number_of_iterations = 10
    for itr in range(number_of_iterations):
        run_jepa(arguments.fname, 'dreamer')

def behavior_learning(arguments):
    training_complete = False
    while not training_complete:
        prepare_dreams()
        training_complete = run_rl(arguments.fname, 'train_agent')

def lifelong_learning(arguments): 
    while True:
        operate_ptz(arguments)
        prepare_images()
        training_complete = run_jepa(arguments.fname, 'world_model')
        training_complete = run_jepa(arguments.fname, 'dreamer')



def main():
    parser = argparse.ArgumentParser("PTZ JEPA")

    # PTZ sampler
    parser.add_argument("-cb", "--camerabrand",
                        help="An integer for each accepted camera brand (default=0). 0 is Hanwha, 1 is Axis.", type=int,
                        default=0)
    parser.add_argument("-it", "--iterations",
                        help="An integer with the number of iterations (PTZ rounds) to be run (default=10).", type=int,
                        default=10)
    parser.add_argument("-mv", "--movements",
                        help="An integer with the number of movements in each PTZ round to be run (default=10).",
                        type=int, default=10)
    parser.add_argument("-un", "--username",
                        help="The username of the PTZ camera.",
                        type=str, default='')
    parser.add_argument("-pw", "--password",
                        help="The password of the PTZ camera.",
                        type=str, default='')
    parser.add_argument("-ip", "--cameraip",
                        help="The ip of the PTZ camera.",
                        type=str, default='')
    parser.add_argument("-rm", "--run_mode",
                        help="The mode to run the code.",
			choices=['train', 'world_model_train', 'dream', 'agent_train', 'lifelong'],
                        type=str, default='train')

    # Joint Embedding Predictive Architecture (JEPA)
    parser.add_argument("-fn", "--fname", type=str,
                        help="name of config file to load",
                        default='/percistence/configs/in1k_vith14_ep300.yaml')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    if args.run_mode=='train':
       pretraining_wrapper(args)
    elif args.run_mode=='world_model_train':
       pretraining_world_model_wrapper(args)
    elif args.run_mode=='dream':
       dreamer_wrapper(args)
    elif args.run_mode=='agent_train':
       behavior_learning(args)
    elif args.run_mode=='lifelong':
       lifelong_learning(args)




    print('DONE!')


if __name__ == "__main__":
    main()
