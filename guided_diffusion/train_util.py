import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy.matlib
from matplotlib.colors import LogNorm
import matplotlib as mpl

from . import logger

if os.uname().nodename == "titan4":
    from guided_diffusion import dist_util_titan as dist_util
else:
    from guided_diffusion import dist_util
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .gaussian_diffusion import _extract_into_tensor
from .nn import mean_flat
from .losses import normal_kl
from .resample import LossAwareSampler, UniformSampler

import wandb

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# INITIAL_LOG_LOSS_SCALE = 20.0
from .unet import UNetModel


class TrainLoop:
    def __init__(
            self,
            *,
            params,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            skip_save,
            save_interval,
            plot_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            scheduler_rate=1,
            scheduler_step=1000,
            num_steps=10000,
            image_size=32,
            in_channels=3,
            class_cond=False,
            max_class=None,
            validator=None,
            validation_interval=None,
            semi_supervised_training=False,
            dims=1
    ):
        self.params = params
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.image_size = image_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.class_cond = class_cond
        self.max_class = max_class
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.skip_save = skip_save
        self.plot_interval = plot_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.num_steps = num_steps

        self.step = 0
        self.resume_step = 0
        self.world_size = dist.get_world_size()
        self.global_batch = self.batch_size * self.world_size

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = th.optim.lr_scheduler.ExponentialLR(self.opt, gamma=scheduler_rate)
        self.scheduler_step = scheduler_step
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.semi_supervised_training = semi_supervised_training
        self.dims = dims  # TODO: Added dims
        if th.cuda.is_available():
            self.use_ddp = False
            find_unused_params = self.diffusion.skip_classifier_loss
            # find_unused_params = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=find_unused_params,
            )
        else:
            if self.world_size > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.validator = validator
        if validator is None:
            self.validation_interval = self.num_steps + 1  # Skipping validation
        else:
            self.validation_interval = validation_interval

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        plot = self.plot
        while (
                (not self.lr_anneal_steps
                 or self.step + self.resume_step < self.lr_anneal_steps) and (self.step < self.num_steps)
        ):
            if self.step > 100:
                self.mp_trainer.skip_gradient_thr = self.params.skip_gradient_thr
            batch, cond = next(self.data)
            self.run_step(batch, cond, self.step)
            if self.step % self.log_interval == 0:
                if logger.get_rank_without_mpi_import() == 0:
                    wandb.log(logger.getkvs(), step=self.step)
                logger.dumpkvs()
            if (not self.skip_save) & (self.step % self.save_interval == 0) & (self.step != 0):
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step > 0:
                if (self.step == 1000) or (self.step % self.plot_interval == 0):
                    plot(self.step)
                if self.step % self.scheduler_step == 0:
                    self.scheduler.step()
                if self.step % self.validation_interval == 0:
                    logger.log(f"Validation for step {self.step}")
                    if self.params.train_with_classifier:
                        preds, test_loss, test_accuracy = self.validator.calculate_accuracy_with_classifier(
                            model=self.model)
                        if logger.get_rank_without_mpi_import() == 0:
                            wandb.log({"preds": preds})
                            wandb.log({"test_classification_loss": test_loss})
                            wandb.log({"test_classification_accuracy": test_accuracy})
                            logger.log(f"Test classification loss: {test_loss}, Test acc: {test_accuracy}")
                    else:
                        fid_result, precision, recall = self.validator.calculate_results(train_loop=self,
                                                                                         dataset=self.params.dataset,
                                                                                         n_generated_examples=self.params.n_examples_validation,
                                                                                         batch_size=self.params.microbatch if self.params.microbatch > 0 else self.params.batch_size)
                        if logger.get_rank_without_mpi_import() == 0:
                            wandb.log({"fid": fid_result})
                            wandb.log({"precision": precision})
                            wandb.log({"recall": recall})
                        logger.log(f"FID: {fid_result}, Prec: {precision}, Rec: {recall}")

            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if not self.skip_save:
            if (self.step - 1) % self.save_interval != 0:
                self.save()
        if (self.step - 1) % self.plot_interval != 0:
            plot(self.step)

    def run_step(self, batch, cond, step):
        self.forward_backward(batch, cond, step)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, step):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            # micro_cond = cond[i: i + self.microbatch].to(dist_util.dev())  # {
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                step=step
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict, suffix=""):
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate} {suffix}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}_0{suffix}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}_0{suffix}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        save_checkpoint(0, state_dict)

        dist.barrier()

    @th.no_grad()
    def generate_examples(self, total_num_exapmles, max_classes, batch_size=-1):
        if batch_size == -1:
            batch_size = total_num_exapmles
        model = self.mp_trainer.model
        model.eval()
        all_images = []
        model_kwargs = {}
        if self.class_cond:
            classes = th.randint(0, max_classes, (total_num_exapmles,1))
            # raise NotImplementedError()
        else:
            classes = None
        i = 0
        while len(all_images) < total_num_exapmles:

            num_examples_to_generate = min(batch_size, total_num_exapmles - len(all_images))
            if self.class_cond:
                model_kwargs["y"] = classes[i * batch_size:i * batch_size + num_examples_to_generate]
            sample_fn = (
                self.diffusion.p_sample_loop  # if not self.use_ddim else diffusion.ddim_sample_loop
            )

            # Adjust shape according to dims
            if self.dims == 1:
                self.image_size = 28
                shape = (num_examples_to_generate, self.in_channels, self.image_size)
            elif self.dims == 2:
                shape = (num_examples_to_generate, self.in_channels, self.image_size, self.image_size)
            else:
                raise ValueError(f"Unsupported dims: {self.dims}")

            sample = sample_fn(
                model,
                shape,
                clip_denoised=False, model_kwargs=model_kwargs,
            )
            # Verify the shape of the generated samples
            assert sample.shape == shape, f"Expected {shape} but got {sample.shape}"

            # sample = sample_fn(
            #     model,
            #     (num_examples_to_generate, self.in_channels, self.image_size, self.image_size),
            #     clip_denoised=False, model_kwargs=model_kwargs,
            # )

            all_images.extend(sample.cpu())
            print(f"generated: {len(all_images)}/{total_num_exapmles}")
            i += 1
        model.train()
        all_images = th.stack(all_images, 0)
        return all_images, classes

    @th.no_grad()
    def plot(self, step, num_examples=8):
        sample, _ = self.generate_examples(num_examples, max_classes=self.max_class)

        # check dims of data
        if self.dims == 1:
            self.plot_1d_samples(sample, step, num_examples)
        elif self.dims == 2:
            self.plot_2d_samples(sample, step, num_examples)
        else:
            raise ValueError(f"Unsupported dims: {self.dims}")

    @th.no_grad()
    def plot_1d_samples(self, sample, step, num_examples):
        print("Plotting samples!")
        print("Num samples: ", sample.size(0))
        print("Num channels: ", sample.size(1))

        num_samples = sample.size(0)
        num_channels = sample.size(1)

        fig, axes = plt.subplots(num_samples, 1, figsize=(15, num_samples * 3), sharex=True)
        if num_samples == 1:
            axes = [axes]  # Make the axes iterable if there's only one sample

        for sample_idx in range(num_samples):
            for channel_idx in range(num_channels):
                axes[sample_idx].plot(sample[sample_idx, channel_idx, :].detach().cpu().numpy(), label=f'Channel {channel_idx + 1}')
            
            axes[sample_idx].set_title(f"Sample {sample_idx + 1}")
            axes[sample_idx].set_xlabel("Time Step")
            if sample_idx == 0:
                axes[sample_idx].legend(loc="upper right")
        
        fig.tight_layout()
        
        # Save plot
        if not os.path.exists(os.path.join(logger.get_dir(), f"samples/")):
            os.makedirs(os.path.join(logger.get_dir(), f"samples/"))
        out_plot = os.path.join(logger.get_dir(), f"samples/step_{step:06d}.png")
        plt.savefig(out_plot)
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log({"sampled_1d_signals": wandb.Image(out_plot)})
        plt.close(fig)


    @th.no_grad()
    def plot_2d_samples(self, sample, step, num_examples):
        samples_grid = make_grid(sample.detach().cpu(), num_examples, normalize=True).permute(1, 2, 0)
        sample_wandb = wandb.Image(samples_grid.permute(2, 0, 1), caption=f"samples")
        logs = {"sampled_images": sample_wandb}

        plt.imshow(samples_grid)
        plt.axis('off')
        if not os.path.exists(os.path.join(logger.get_dir(), f"samples/")):
            os.makedirs(os.path.join(logger.get_dir(), f"samples/"))
        out_plot = os.path.join(logger.get_dir(), f"samples/step_{step:06d}")
        plt.savefig(out_plot)
        if logger.get_rank_without_mpi_import() == 0:
            wandb.log(logs, step=step)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if key != "prev_kl":
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                if sub_t == 0:
                    logger.logkv_mean(f"{key}_0_step", sub_loss)


def time_hist(ax, data):
    num_pt, num_ts = data.shape
    num_fine = num_pt * 10
    x_fine = np.linspace(0, num_pt, num_fine)
    y_fine = np.empty((num_ts, num_fine), dtype=float)
    for i in range(num_ts):
        y_fine[i, :] = np.interp(x_fine, range(num_pt), data[:, i])
    y_fine = y_fine.flatten()
    x_fine = np.matlib.repmat(x_fine, num_ts, 1).flatten()
    cmap = copy.copy(plt.cm.BuPu)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[40, 1000])
    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, vmax=num_ts, rasterized=True)
    plt.colorbar(pcm, ax=ax, label="# points", pad=0);
