import sys
import logging
import copy
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import numpy as np
from argparse import ArgumentParser
from helper import init_model
from run_jepa import (
    get_position_from_label,
    forward_context,
    arrange_inputs,
    forward_target,
)
from transforms import make_transforms
from datasets.ptz_dataset import PTZImageDataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("gen_embed")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-conf",
        "--config_fpath",
        type=str,
        default="configs/ptz.yaml",
        help="path to the config file",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_fpath",
        type=str,
        default="checkpoints/ptz.pth",
        help="path to the checkpoint file",
    )
    parser.add_argument(
        "-img",
        "--img_dir",
        type=str,
        help="path to the directory containing images",
    )
    parser.add_argument(
        "-out",
        "--output_dir",
        type=str,
        help="path to the output directory",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="device to run the model on",
    )
    parser.add_argument(
        "-r",
        '--remove_corrupt',
        action='store_true',
        help='Remove corrupt images from the directory'
    )
    return parser.parse_args()


def check_file_integrity(dir_path, remove_corrupt=False):
    """
    Check the integrity of image files in a directory and optionally remove corrupt files.

    Args:
        dir_path (str): The path to the directory containing the image files.
        remove_corrupt (bool, optional): Whether to remove corrupt files. Defaults to False.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.

    Returns:
        None
    """
    from PIL import Image
    
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"{dir_path} does not exist")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory")
    for fp in dir_path.glob("*.jpg"):
        try:
            image = Image.open(fp)
            image.verify()
            # need to reopen to save the image again
            image = Image.open(fp)
            image.save(fp) # to fix corrupted images
        except (OSError, IOError) as e:
            logger.error("Got error: %s", e)
            logger.error("Corrupted image: %s", fp)
            if remove_corrupt:
                fp.unlink()


def generate_embedding(
    config_fpath, checkpoint_fpath: str, img_dir: str, output_dir: str, device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    img_dir = Path(img_dir)
    if not output_dir.exists():
        logger.info("Creating output directory")
        output_dir.mkdir()
    logger.info("Loading parameters")
    with open(config_fpath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # use_bfloat16 = config["meta"]["use_bfloat16"]
    model_name = config["meta"]["model_name"]
    pred_depth = config["meta"]["pred_depth"]
    pred_emb_dim = config["meta"]["pred_emb_dim"]
    patch_size = config["mask"]["patch_size"]  # patch-size for model training
    crop_size = config["data"]["crop_size"]
    batch_size = config["data"]["batch_size"]
    camera_brand = config["meta"]["camera_brand"]

    # load model
    logger.info("Loading model")
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
    )
    target_encoder = copy.deepcopy(encoder)
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    pretrained_dict = checkpoint["encoder"]
    msg = encoder.load_state_dict(pretrained_dict)
    logger.info("loaded context encoder from epoch %s with msg: %s", epoch, msg)
    # pretrained_dict = checkpoint["predictor"]
    # msg = predictor.load_state_dict(pretrained_dict)
    # logger.info("loaded predictor from epoch %s with msg: %s", epoch, msg)
    pretrained_dict = checkpoint["target_encoder"]
    msg = target_encoder.load_state_dict(pretrained_dict)
    logger.info("loaded target encoder from epoch %s with msg: %s", epoch, msg)

    # load dataset
    logger.info("Loading dataset")
    data = PTZImageDataset(img_dir, transform=make_transforms())
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ipe = len(dataloader)

    # start inference
    logger.info("Starting inference")
    labels = []
    contx_encoder_embed = []
    # predictor_embed = []
    target_encoder_embed = []
    for i, (img, label) in enumerate(dataloader):
        pos = get_position_from_label(label)
        img = img.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        # context_imgs, context_poss, target_imgs, target_poss = arrange_inputs(
        #     img, pos, device
        # )
        with torch.no_grad():
            # contx_enc_embed, contx_pred_embed = forward_context(
            #     context_imgs,
            #     context_poss,
            #     target_poss,
            #     encoder,
            #     predictor,
            #     camera_brand,
            #     return_all_embeddings=True,
            # )
            contx_enc_embed = encoder.forward(img)
            tar_embed = forward_target(img, target_encoder)
        contx_encoder_embed.append(contx_enc_embed.cpu().numpy())
        # predictor_embed.append(contx_pred_embed.cpu().numpy())
        target_encoder_embed.append(tar_embed.cpu().numpy())
        labels += list(label)
        if i and i % 100 == 0:
            logger.info("Processed %d/%d", i, ipe)
            save_data = lambda arr, fpath: np.vstack(np.array(arr)).dump(fpath)
            save_data(contx_encoder_embed, output_dir / "embeds_contx_encoder.npy")
            # save_data(predictor_embed, output_dir / "embeds_predictor.npy")
            save_data(target_encoder_embed, output_dir / "embeds_target_encoder.npy")
            with open(output_dir / "labels.txt", "w") as fp:
                fp.write("\n".join(labels))


if __name__ == "__main__":
    args = parse_args()
    check_file_integrity(args.img_dir, remove_corrupt=args.remove_corrupt)
    generate_embedding(
        args.config_fpath,
        args.checkpoint_fpath,
        args.img_dir,
        args.output_dir,
        args.device
    )
    
