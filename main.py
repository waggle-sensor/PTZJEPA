# import sys
# sys.path.append("/app/source")

import os
import logging
import argparse

from source.prepare_dataset import get_images_from_storage, prepare_images, operate_ptz
from source.run_jepa import run as run_jepa
from source.run_rl import run as run_rl
from source.env_interaction import run as env_inter


logger = logging.getLogger(__name__)


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
        # prepare_dreams()
        training_complete = run_rl(arguments.fname, "train_agent")


def environment_interaction(arguments):
    interaction_complete = env_inter(arguments, arguments.fname, "navigate_env")
    logger.info("interaction_complete: %s", interaction_complete)


def lifelong_learning(arguments):
    operate_ptz(arguments)
    while True:
        prepare_images()
        training_complete = run_jepa(arguments.fname, "world_model")
        #if training_complete:
            #continue
        training_complete = run_jepa(arguments.fname, "dreamer")
        training_complete = run_rl(arguments.fname, "train_agent")
        #if training_complete:
            #continue
        interaction_complete = env_inter(arguments, arguments.fname, "navigate_env")


def get_argparser():
    parser = argparse.ArgumentParser("PTZ JEPA")
    # PTZ sampler
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-tp",
        "--trackpositions",
        action="store_true",
        help="Track camera positions storing them in persistent folder for later analysis",
    )
    parser.add_argument(
        "--track_all",
        action="store_true",
        help="Keep positions, commands and embeddings in persistent folder",
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the program in debug mode with more verbose output info",
    )

    parser.add_argument(
        "-pubmsg",
        "--publish_msgs",
        action="store_true",
        help="Whether to publish messages via waggle plugin",
    )
    return parser


def main():
    args = get_argparser().parse_args()
    log_level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=log_level)
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

    logger.info("DONE!")


if __name__ == "__main__":
    main()
