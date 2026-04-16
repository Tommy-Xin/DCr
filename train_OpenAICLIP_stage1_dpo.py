import argparse
import logging
import math
import os
import re
from copy import deepcopy

import datasets
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm.auto import tqdm

from clip_models.build_CLIP import load_clip_model_OpenAICLIP
from clip_models.sampling import prepare_clip
from image_datasets.dataset_cc3m import loader
from src.stable_diffusion import build_sd_model

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
VAE_MEAN = 0.5
VAE_STD = 0.5
NORMALIZE_CLIP = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
NORMALIZE_VAE = transforms.Normalize(mean=VAE_MEAN, std=VAE_STD)


class SuperModel(nn.Module):
    def __init__(self, clip_vis, dit):
        super().__init__()
        self.clip_vis = clip_vis
        self.dit = dit

    def get_clip_vis(self):
        return self.clip_vis

    def get_dit(self):
        return self.dit


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 training with DPO objective.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


def compute_dpo_loss(anchor_vec, pos_plus_vec, gt_vec, all_vec, tau=0.1, dpo_beta=1.0):
    """
    Implements:
    max E[ log sigma( beta * (sum_{p in P} log d - sum_{n in N} log d) / sum_{x in P∪N} log d ) ]

    Here we use log d = scaled cosine logit:
        log d(a, b) = <a, b> / tau
    and construct negatives separately from:
        - anchor vs batch candidates
        - gt vs batch candidates
    """
    logits_all_anchor = torch.einsum("bd,bkd->bk", anchor_vec, all_vec) / tau
    logits_all_gt = torch.einsum("bd,bkd->bk", gt_vec, all_vec) / tau

    bs = anchor_vec.shape[0]
    neg_mask = ~torch.eye(bs, dtype=torch.bool, device=anchor_vec.device)

    logits_pos_pred = (anchor_vec * pos_plus_vec).sum(dim=1) / tau
    logits_pos_gt = (anchor_vec * gt_vec).sum(dim=1) / tau

    # P = {pos_plus, gt}
    pos_log_sum = logits_pos_pred + logits_pos_gt

    # N is constructed from two separately generated negative pools.
    neg_anchor_logs = logits_all_anchor[neg_mask].view(bs, bs - 1)
    neg_gt_logs = logits_all_gt[neg_mask].view(bs, bs - 1)
    neg_log_sum = neg_anchor_logs.sum(dim=1) + neg_gt_logs.sum(dim=1)

    total_log_sum = pos_log_sum + neg_log_sum
    eps = 1e-6
    denom = torch.where(total_log_sum.abs() < eps, total_log_sum.sign() * eps + eps, total_log_sum)

    preference = (pos_log_sum - neg_log_sum) / denom
    loss = -F.logsigmoid(dpo_beta * preference)
    return loss, pos_log_sum, neg_log_sum, denom, preference


def main():
    args = OmegaConf.load(parse_args())
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    sd_parts = build_sd_model(
        model_name=args.clip_config.sd_model_name,
        device=accelerator.device,
        dtype=torch.bfloat16,
    )
    vae = sd_parts["vae"]
    unet = sd_parts["unet"]
    clip_vis = load_clip_model_OpenAICLIP(args.clip_config, device=accelerator.device)

    if args.clip_config.clip_image_size == 336:
        clip_vis.model.visual_projection.weight = torch.nn.Parameter(clip_vis.model.visual_projection.weight.contiguous())
        clip_vis.model.text_projection.weight = torch.nn.Parameter(clip_vis.model.text_projection.weight.contiguous())

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet = unet.to(torch.bfloat16)
    unet.to(accelerator.device)
    clip_vis.train()
    unet.train()

    for name_, param in clip_vis.named_parameters():
        if "project_clip" in name_:
            param.requires_grad = True
        else:
            param.requires_grad = False

    super_model = SuperModel(clip_vis, unet)
    optimizer = torch.optim.AdamW(
        [p for p in super_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(int(3e6) / args.data_config.train_batch_size) / args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    super_model, optimizer, _, lr_scheduler = accelerator.prepare(
        super_model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    dpo_beta = float(getattr(args, "dpo_beta", 1.0))
    tau = float(getattr(args, "tau", 0.1))

    logger.info("***** Running training (DPO) *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  DPO beta = {dpo_beta}, tau = {tau}")

    def save_resume_checkpoint(unwrapped_super_model, optimizer, lr_scheduler, global_step, epoch):
        save_path_resume = os.path.join(args.output_dir, f"checkpoint-resume-{global_step}.pt")
        tmp_save_path_resume = f"{save_path_resume}.tmp"
        resume_state = {
            "global_step": global_step,
            "epoch": epoch,
            "super_model": deepcopy(unwrapped_super_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(resume_state, tmp_save_path_resume)
        os.replace(tmp_save_path_resume, save_path_resume)
        return save_path_resume

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            candidates = [os.path.basename(args.resume_from_checkpoint)]
        else:
            entries = os.listdir(args.output_dir)
            dirs = [d for d in entries if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            pt_files = [d for d in entries if d.startswith("checkpoint-resume-") and d.endswith(".pt")]

            def _extract_step(name):
                match = re.search(r"(\d+)", name)
                return int(match.group(1)) if match else -1

            candidates = sorted(dirs + pt_files, key=lambda x: _extract_step(x), reverse=True)

        if len(candidates) == 0:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            last_error = None
            resumed = False
            for path in candidates:
                resume_path = os.path.join(args.output_dir, path)
                accelerator.print(f"Trying to resume from checkpoint {path}")
                try:
                    if path.endswith(".pt"):
                        checkpoint = torch.load(resume_path, map_location="cpu")
                        unwrapped_super_model = accelerator.unwrap_model(super_model)
                        unwrapped_super_model.load_state_dict(checkpoint["super_model"], strict=True)
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                        global_step = int(checkpoint["global_step"])
                        first_epoch = int(checkpoint.get("epoch", global_step // num_update_steps_per_epoch))
                    else:
                        accelerator.load_state(resume_path)
                        global_step = int(path.split("-")[1])
                        first_epoch = global_step // num_update_steps_per_epoch
                    resumed = True
                    accelerator.print(f"Resumed from checkpoint {path}")
                    break
                except Exception as err:
                    last_error = err
                    logger.warning(f"Failed loading checkpoint {resume_path}: {repr(err)}")

            if resumed:
                initial_global_step = global_step
            else:
                accelerator.print(
                    f"All candidate checkpoints failed to load. Starting a new training run. Last error: {repr(last_error)}"
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(super_model):
                original_img, _ = batch["image"], batch["text"]
                original_img = original_img.to(accelerator.device)

                with torch.no_grad():
                    x_1 = vae.encode(
                        NORMALIZE_VAE(original_img).to(device=vae.device, dtype=weight_dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                inp = prepare_clip(
                    clip=clip_vis,
                    original_img=NORMALIZE_CLIP(original_img).to(weight_dtype),
                    img=x_1.to(weight_dtype),
                )
                bs = original_img.shape[0]
                t = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=accelerator.device).long()
                noise = torch.randn_like(x_1).to(accelerator.device)
                noisy_latents = noise_scheduler.add_noise(x_1, noise, t)

                model_pred = unet(
                    noisy_latents[:, None].repeat(1, bs, 1, 1, 1).reshape(bs * bs, *noisy_latents.shape[1:]),
                    t[:, None].repeat(1, bs).reshape(bs * bs),
                    encoder_hidden_states=inp["vec"].to(weight_dtype)[None]
                    .repeat(bs, 1, 1, 1)
                    .reshape(bs * bs, inp["vec"].shape[1], inp["vec"].shape[2]),
                ).sample.reshape(bs, bs, *noisy_latents.shape[1:])

                model_pred_plus = unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=inp["vec_plus"].to(weight_dtype),
                ).sample

                target_v = noise_scheduler.get_velocity(x_1, noise, t)

                anchor_pred = model_pred[torch.arange(bs), torch.arange(bs)]
                anchor_vec = F.normalize(anchor_pred.flatten(1), dim=1)
                gt_vec = F.normalize(target_v.flatten(1), dim=1)
                pos_plus_vec = F.normalize(model_pred_plus.flatten(1), dim=1)
                all_vec = F.normalize(model_pred.reshape(bs, bs, -1), dim=2)

                # DPO objective with separately aggregated negative samples.
                loss, pos_log_sum, neg_log_sum, denom, preference = compute_dpo_loss(
                    anchor_vec=anchor_vec,
                    pos_plus_vec=pos_plus_vec,
                    gt_vec=gt_vec,
                    all_vec=all_vec,
                    tau=tau,
                    dpo_beta=dpo_beta,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(super_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    unwrapped_super_model = accelerator.unwrap_model(super_model)
                    save_path_dit = os.path.join(args.output_dir, f"checkpoint-unet-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                    save_path_project_clip = os.path.join(
                        args.output_dir, f"checkpoint-project-clip-{global_step}.bin"
                    )
                    torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                    save_path_optimizer = os.path.join(args.output_dir, f"optimizer-state-{global_step}.bin")
                    torch.save(optimizer.state_dict(), save_path_optimizer)
                    save_path_resume = save_resume_checkpoint(
                        unwrapped_super_model=unwrapped_super_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        epoch=epoch,
                    )
                    logger.info(
                        f"Saved ckpts to {save_path_dit}, {save_path_project_clip}, and resumable {save_path_resume}"
                    )

            logs = {"step_loss": loss.detach().item(), "pos_log_sum": pos_log_sum.detach().mean().item(), "neg_log_sum": neg_log_sum.detach().mean().item(), "denom": denom.detach().mean().item(), "preference": preference.detach().mean().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    unwrapped_super_model = accelerator.unwrap_model(super_model)
                    save_path_dit = os.path.join(args.output_dir, f"checkpoint-unet-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.dit).state_dict(), save_path_dit)
                    save_path_project_clip = os.path.join(args.output_dir, f"checkpoint-project-clip-{global_step}.bin")
                    torch.save(deepcopy(unwrapped_super_model.clip_vis.project_clip).state_dict(), save_path_project_clip)
                    save_resume_checkpoint(
                        unwrapped_super_model=unwrapped_super_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        epoch=epoch,
                    )
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
