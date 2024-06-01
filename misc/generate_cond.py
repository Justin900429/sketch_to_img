import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import numpy_to_pil
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from config import create_cfg, merge_possible_with_base, show_config
from modeling import build_model

SCHEDULER_FUNC = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "dpm": DPMSolverMultistepScheduler,
}

OUT_WIDTH = 448
OUT_HEIGHT = 240


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
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None, type=str)
    return parser.parse_args()


@torch.no_grad()
def evaluate(cfg, model, noise_scheduler, conditional_imgs, labels, device, num_images):
    model.eval()
    image_shape = (
        num_images,
        3,
        cfg.TRAIN.IMAGE_SIZE,
        cfg.TRAIN.IMAGE_SIZE,
    )
    images = torch.randn(image_shape, device=device)
    conditional_imgs = conditional_imgs.to(device)
    labels = labels.to(device)

    noise_scheduler.set_timesteps(cfg.EVAL.SAMPLE_STEPS, device=device)
    for t in tqdm(noise_scheduler.timesteps):
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
        images = noise_scheduler.step(model_output, t, images).prev_sample

    images = (images.to(torch.float32).clamp(-1, 1) + 1) / 2
    images = numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())
    return images


def generate_image(cfg, save_folder, image_list, device, batch_size: int = 64):
    noise_scheduler = SCHEDULER_FUNC[cfg.EVAL.SCHEDULER](
        num_train_timesteps=cfg.TRAIN.SAMPLE_STEPS,
        prediction_type=cfg.TRAIN.NOISE_SCHEDULER.PRED_TYPE,
        beta_schedule=cfg.TRAIN.NOISE_SCHEDULER.TYPE,
        # For linear only
        beta_start=cfg.TRAIN.NOISE_SCHEDULER.BETA_START,
        beta_end=cfg.TRAIN.NOISE_SCHEDULER.BETA_END,
        thresholding=True,
    )
    model = build_model(cfg, train=False)
    model.to(device)

    conditioning_img_transforms = T.Compose(
        [
            T.Resize(
                (cfg.TRAIN.IMAGE_SIZE, cfg.TRAIN.IMAGE_SIZE),
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
        print("weights are laoded")

    os.makedirs(args.save_folder, exist_ok=True)

    model.eval()
    progress_bar = tqdm(total=len(image_list), position=int(device.split(":")[-1]))

    for idx in range(0, len(image_list), batch_size):
        num_images = min(batch_size, len(image_list) - idx)
        end_idx = min(idx + batch_size, len(image_list))

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

        # ========= Build label =========
        labels = torch.LongTensor(
            [(0 if "RO" in img_name else 1) for img_name in image_list[idx:end_idx]]
        )
        labels = torch.nn.functional.one_hot(labels, num_classes=2)
        non_labels = torch.zeros_like(labels)
        labels = torch.cat([non_labels, non_labels, labels], dim=0)

        samples = evaluate(
            cfg, model, noise_scheduler, conditional_imgs, labels, device, num_images
        )

        for num, sample in enumerate(samples):
            progress_bar.update(1)
            sample.resize((OUT_WIDTH, OUT_HEIGHT)).save(
                os.path.join(
                    save_folder,
                    f"{os.path.basename(image_list[idx + num]).replace('png', 'jpg')}",
                )
            )
    progress_bar.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config:
        merge_possible_with_base(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    show_config(cfg)

    image_list = sorted(list(glob.glob(os.path.join(args.cond_root, "*.png"))))
    task_per_process = len(image_list) // args.num_process
    Parallel(n_jobs=args.num_process)(
        delayed(generate_image)(
            cfg,
            args.save_folder,
            image_list[
                gpu_idx * task_per_process : min((gpu_idx + 1) * task_per_process, len(image_list))
            ],
            f"cuda:{gpu_idx}",
            batch_size=args.batch_size,
        )
        for gpu_idx in range(args.num_process)
    )
