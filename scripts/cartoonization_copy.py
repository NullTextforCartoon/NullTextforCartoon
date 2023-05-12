from safetensors import safe_open
from safetensors.torch import save_file

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_model_from_config_cartoon(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt[-4:] != 'ckpt':
        sd = {}
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    else:

        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def get_ref_img(model,image_path, batch_size, device):
    ################add ref image###############
    # image_path = 'img/ref.png'
    image = np.array(Image.open(image_path).resize((512, 512)))[:, :, :3]
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device).repeat(batch_size, 1, 1, 1)
    # image = torch.cat((image0,image),dim=0)
    ref_img = model.encode_first_stage(image)
    ref_img = model.get_first_stage_encoding(ref_img)
    return ref_img

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="mode 0,1,2"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/0412"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/public/data0/MULT/users/zhengheliang/code/stable_diffusion/Dreambooth_zj_cartoon/sd-v1-4-full-ema.ckpt",
        # default="sd-v1-4-full-ema.ckpt",
        # default="sd-v1-4-full-ema.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to a pre-trained embedding manager checkpoint")

    parser.add_argument(
        "--image_path",
        type=str,
        default='Corgi04.png',
    )

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # model0 = model0.to(device)

    batch_size = opt.n_samples
    seed = 796
    merge_mode = opt.mode
    prompt = opt.prompt

    prompt_name = prompt.split(",")[0].replace(" ", "_")
    data = [batch_size * [prompt]]
    # image_name = f"{image_type}0{i}" if i < 10 else f"{image_type}{i}"
    image_path = opt.image_path
    ref_img = None
    if merge_mode in [2, 3]:
        if image_path == None:
            print("Need to enter the image path!")
            return
        ref_img = get_ref_img(model, image_path, batch_size, device)
    outpath = f"ouput/cartoonization/mode{merge_mode}/{prompt_name}"
    os.makedirs(outpath, exist_ok=True)

    def cartoonization(start_t,B,ddim_steps,merge_mode):
        sampler = DDIMSampler(model, merge_mode=merge_mode)
        back_t = int((B * ddim_steps) / 1000)
        mykwargs = {"start_t": start_t, "back_t": back_t}
        outimg_name = f"start{start_t}-back{B}-step{ddim_steps}"
        seed_everything(seed)
        base_count = len(os.listdir(outpath))
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c,
                                                             batch_size=batch_size,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=opt.scale,
                                                             unconditional_conditioning=uc,
                                                             eta=opt.ddim_eta,
                                                             x_T=start_code,
                                                             ref_img=ref_img,
                                                             mykwargs=mykwargs)
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                sample_path = os.path.join(outpath, outimg_name)
                                os.makedirs(sample_path, exist_ok=True)
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{base_count:05}.jpg"))
                                    base_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_samples_ddim)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        summary_grid = grid
                        grid = make_grid(grid, nrow=batch_size)
                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{outimg_name}.jpg'))

        print(f"Your samples are ready and waiting for you here: \n{outpath}/{outimg_name} \n")
        return summary_grid

    start_ts = [200,300,400]
    Bs = [100,200,300] if merge_mode in [1,2] else [100]
    ddim_steps = 100

    if merge_mode == 1:
        _ = cartoonization(0, 0, ddim_steps, 0)
    summary = []
    for B in Bs:
        for start_t in start_ts:
            cartoon_img = cartoonization(start_t, B, ddim_steps, merge_mode)
            summary.append(cartoon_img)
    results = torch.stack(summary, 0)
    results = rearrange(results, 'n b c w h -> b n c w h')
    # results_name = f'start{start_t}-B{B}-step{ddim_steps}'
    idex = 0
    # os.makedirs(outpath, exist_ok=True)
    for sheet in results:
        sheet = make_grid(sheet, nrow=len(start_ts))
        # sheet = make_grid(sheet, nrow=1)
        sheet = 255. * rearrange(sheet, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(sheet.astype(np.uint8)).save(os.path.join(outpath, f'summary_{idex}.jpg'))
        idex += 1


if __name__ == "__main__":
    main()
