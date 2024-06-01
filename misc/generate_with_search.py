import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random

import cv2
import torch
import torchvision.transforms as T
from diffusers.schedulers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import numpy_to_pil
from joblib import Parallel, delayed
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from config import create_cfg, merge_possible_with_base, show_config
from modeling import build_model

INVERSE_SCHEDULER_FUNC = {
    "ddim": DDIMInverseScheduler,
    "dpm": DPMSolverMultistepInverseScheduler,
}
SCHEDULER_FUNC = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "dpm": DPMSolverMultistepScheduler,
}

OUT_WIDTH = 428
OUT_HEIGHT = 240


def generate_great_ssim(img_name, ori_list):
    src_img = cv2.imread(img_name, 0)
    max_score = -1
    copy_name = None
    for ori_name in ori_list:
        ori_img = cv2.imread(ori_name, 0)
        score = structural_similarity(src_img, ori_img)
        if score > max_score:
            max_score = score
            copy_name = ori_name
    return copy_name


def copy_parameters(from_parameters: torch.nn.Parameter, to_parameters: torch.nn.Parameter):
    to_parameters = list(to_parameters)
    assert len(from_parameters) == len(to_parameters)
    for s_param, param in zip(from_parameters, to_parameters):
        param.data.copy_(s_param.to(param.device).data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond-root", required=True, type=str)
    parser.add_argument("--save-folder", default="generate_output", type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-process", default=1, type=int)
    parser.add_argument("--use-ssim", action="store_true", default=False)
    parser.add_argument("--add-noise-step", default=0, type=int)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


@torch.no_grad()
def inverse(cfg, model, images, inverse_scheduler, conditional_imgs, labels, device):
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    conditional_imgs = conditional_imgs.to(device)
    inverse_scheduler.set_timesteps(cfg.EVAL.SAMPLE_STEPS, device=device)
    for t in tqdm(inverse_scheduler.timesteps):
        stack_images = torch.cat([images, images, images], dim=0)
        stack_images = torch.cat((stack_images, conditional_imgs), dim=1)
        (model_output_uncond, model_output_cond_img, model_output_cond_label) = (
            model(stack_images, t.reshape(-1), labels)[:, :-1].float().chunk(3, dim=0)
        )
        model_output = (
            model_output_uncond
            + cfg.EVAL.GUIDANCE * (model_output_cond_img - model_output_uncond) / 2
            + cfg.EVAL.GUIDANCE * (model_output_cond_label - model_output_uncond) / 2
        )
        if cfg.EVAL.SCHEDULER == "ddim":
            images = inverse_scheduler.step(
                model_output, t, images, use_clipped_model_output=True
            ).prev_sample
        else:
            images = inverse_scheduler.step(model_output, t, images).prev_sample
    return images


@torch.no_grad()
def evaluate(
    cfg,
    model,
    noise_scheduler,
    conditional_imgs,
    labels,
    device,
    num_images=None,
    images=None,
    add_noise_step=0,
):
    model.eval()
    if images is None:
        image_shape = (num_images, 3, cfg.TRAIN.IMAGE_SIZE_HEIGHT, cfg.TRAIN.IMAGE_SIZE_WIDTH)
        images = torch.randn(image_shape, device=device)

    conditional_imgs = conditional_imgs.to(device)
    labels = labels.to(device)

    if add_noise_step > 0:
        noise = torch.randn_like(images)
        images = noise_scheduler.add_noise(
            images, noise, torch.tensor([add_noise_step], device=device)
        )
    images = images.to(device)

    noise_scheduler.set_timesteps(cfg.EVAL.SAMPLE_STEPS, device=device)
    for t in tqdm(noise_scheduler.timesteps[-add_noise_step:]):
        stack_images = torch.cat([images, images, images], dim=0)
        stack_images = torch.cat((stack_images, conditional_imgs), dim=1)
        (model_output_uncond, model_output_cond_img, model_output_cond_label) = (
            model(stack_images, t.reshape(-1), labels)[:, :-1].float().chunk(3, dim=0)
        )
        model_output = (
            model_output_uncond
            + cfg.EVAL.GUIDANCE * (model_output_cond_img - model_output_uncond) / 2
            + cfg.EVAL.GUIDANCE * (model_output_cond_label - model_output_uncond) / 2
        )
        if cfg.EVAL.SCHEDULER == "ddim":
            images = noise_scheduler.step(
                model_output, t, images, use_clipped_model_output=True, eta=cfg.EVAL.ETA
            ).prev_sample
        else:
            images = noise_scheduler.step(model_output, t, images).prev_sample

    images = (images.to(torch.float32).clamp(-1, 1) + 1) / 2
    images = numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())
    return images


def generate_image(
    cfg, save_folder, image_list, device, random_img_list=None, batch_size: int = 64
):
    torch.cuda.set_device(device)
    if args.add_noise_step == 0:
        inverse_scheduler = INVERSE_SCHEDULER_FUNC[cfg.EVAL.SCHEDULER](
            num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
            prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
            beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
            # For linear only
            beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
            beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
            thresholding=True,
            lambda_min_clipped=-5.1,
        )

    noise_scheduler = SCHEDULER_FUNC[cfg.EVAL.SCHEDULER](
        num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
        prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
        beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
        # For linear only
        beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
        beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
        thresholding=True,
        lambda_min_clipped=-5.1,
    )
    model = build_model(cfg, use_free_U=False).to(device)

    img_transforms = T.Compose(
        [
            T.Resize(
                (cfg.TRAIN.IMAGE_SIZE_HEIGHT, cfg.TRAIN.IMAGE_SIZE_WIDTH),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_img_transforms = T.Compose(
        [
            T.Resize(
                (cfg.TRAIN.IMAGE_SIZE_HEIGHT, cfg.TRAIN.IMAGE_SIZE_WIDTH),
                interpolation=T.InterpolationMode.NEAREST,
            ),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    if cfg.EVAL.CHECKPOINT:
        weight = torch.load(cfg.EVAL.CHECKPOINT, map_location=device)
        model.load_state_dict(weight["state_dict"])
        copy_parameters(weight["ema_state_dict"]["shadow_params"], model.parameters())
        del weight
        torch.cuda.empty_cache()
        print("weights are loaded")

    model.eval()
    progress_bar = tqdm(total=len(image_list), position=int(device.split(":")[-1]))

    if args.use_ssim:
        if os.path.exists("save.pkl"):
            with open("save.pkl", "rb") as f:
                save_label_res = pickle.load(f)
        else:
            save_label_res = {}

    for idx in range(0, len(image_list), batch_size):
        end_idx = min(len(image_list), idx + batch_size)

        # ========= Build label =========
        labels = torch.LongTensor(
            [(0 if "RO" in img_name else 1) for img_name in image_list[idx:end_idx]]
        )
        labels = torch.nn.functional.one_hot(labels, num_classes=2)
        non_labels = torch.zeros_like(labels)
        labels = torch.cat([non_labels, non_labels, labels], dim=0)

        if args.use_ssim:
            label_to_inverse = []
            for img in image_list[idx:end_idx]:
                if img not in save_label_res:
                    save_label_res[img] = generate_great_ssim(
                        img, ori_road if "RO" in img else ori_river
                    )
                label_to_inverse.append(save_label_res[img])
        else:
            label_to_inverse = random_img_list[idx:end_idx]

        image_to_inverse = torch.stack(
            [
                img_transforms(
                    Image.open(img.replace("label_img", "img").replace("png", "jpg")).convert("RGB")
                )
                for img in label_to_inverse
            ],
            dim=0,
        )
        if args.add_noise_step == 0:
            label_to_inverse = torch.stack(
                [
                    conditioning_img_transforms(Image.open(img).convert("L"))
                    for img in label_to_inverse
                ],
                dim=0,
            )
            non_label_to_inverse = torch.zeros_like(label_to_inverse)
            label_to_inverse = torch.cat(
                [non_label_to_inverse, label_to_inverse, non_label_to_inverse], dim=0
            )
            inverse_samples = inverse(
                cfg,
                model,
                image_to_inverse,
                inverse_scheduler,
                label_to_inverse,
                labels,
                device,
            )

        # ========= Build sketch conditional image ==========
        conditional_imgs = [
            conditioning_img_transforms(Image.open(img).convert("L"))
            for img in image_list[idx:end_idx]
        ]
        conditional_imgs = torch.stack(conditional_imgs, dim=0)
        non_conditonal_imgs = torch.zeros_like(conditional_imgs)
        conditional_imgs = torch.cat(
            [non_conditonal_imgs, conditional_imgs, non_conditonal_imgs], dim=0
        )

        samples = evaluate(
            cfg,
            model,
            noise_scheduler,
            conditional_imgs,
            labels,
            device,
            num_images=len(image_list[idx:end_idx]),
            add_noise_step=args.add_noise_step,
            images=inverse_samples if args.add_noise_step == 0 else image_to_inverse,
        )

        for img_name, sample in zip(image_list[idx:end_idx], samples):
            sample.resize((OUT_WIDTH, OUT_HEIGHT)).save(
                os.path.join(
                    save_folder,
                    f"{os.path.basename(img_name).replace('png', 'jpg')}",
                )
            )
            progress_bar.update(1)
    progress_bar.close()

    if args.use_ssim:
        if not os.path.exists("save.pkl"):
            with open("save.pkl", "wb") as f:
                pickle.dump(save_label_res, f)


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config:
        merge_possible_with_base(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    show_config(cfg)
    os.makedirs(args.save_folder, exist_ok=True)

    image_list = sorted(list(glob.glob(os.path.join(args.cond_root, "*.png"))))
    image_list = [
        img
        for img in image_list
        if not os.path.exists(
            os.path.join(args.save_folder, f"{os.path.basename(img).replace('png', 'jpg')}")
        )
    ]
    image_list.sort()
    count_road = sum(["RO" in os.path.basename(img) for img in image_list])
    count_river = sum(["RO" not in os.path.basename(img) for img in image_list])
    ori_label_data = glob.glob("data/label_img/*.png")
    ori_road = [img for img in ori_label_data if "RO" in img]
    ori_river = [img for img in ori_label_data if "RO" not in img]
    random.shuffle(ori_road)
    random.shuffle(ori_river)
    random_img_list = [ori_road.pop() if "RO" in img else ori_river.pop() for img in image_list]

    task_per_process = len(image_list) // args.num_process
    Parallel(n_jobs=args.num_process)(
        delayed(generate_image)(
            cfg,
            args.save_folder,
            image_list[
                gpu_idx * task_per_process : min((gpu_idx + 1) * task_per_process, len(image_list))
            ],
            f"cuda:{gpu_idx}",
            random_img_list[
                gpu_idx * task_per_process : min((gpu_idx + 1) * task_per_process, len(image_list))
            ],
            batch_size=args.batch_size,
        )
        for gpu_idx in range(args.num_process)
    )
