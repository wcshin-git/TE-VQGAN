import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import importlib

from taming.modules.diffusionmodules.model import Encoder, Decoder, VUNet
from taming.modules.vqvae.quantize import VectorQuantizer

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from itertools import combinations
import math

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ortho_reg_coef=0.0,
                 force_emb_len_1=False,
                 distance_measure='L2',
                 eq_metric_dataset='mnist',
                 ):
        super().__init__()
        print("Orthogonality Regularization Coef:", ortho_reg_coef)
        print("force_emb_len_1:", force_emb_len_1)
        print("distance_measure:", distance_measure)
        print("eq metric dataset:", eq_metric_dataset)
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, ortho_reg_coef=ortho_reg_coef, force_emb_len_1=force_emb_len_1, distance_measure=distance_measure)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.eq_metric_dataset = eq_metric_dataset
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, emb_loss, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, emb_loss

    def get_input(self, batch, k):
        if self.eq_metric_dataset == 'imagenet':
            x = batch[0]
            x = (x * 2 - 1.0)
            return x
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, emb_loss = self(x)
        self.log("train/ortho_reg_term", emb_loss[1])
        qloss = emb_loss[0] + emb_loss[1]
        
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        if self.eq_metric_dataset in ['mnist', 'bird', 'fashion'] and (batch_idx == 0 and self.global_rank == 0):
            if self.eq_metric_dataset == 'mnist':
                Metric = ShiftEqMetric(root='datasets/MNIST64x64_Translated', batch_size=100)
            elif self.eq_metric_dataset == 'bird':
                Metric = ShiftEqMetric(root='datasets/BIRD256x256_Translated', batch_size=15)
            elif self.eq_metric_dataset == 'fashion':
                Metric = ShiftEqMetric(root='datasets/FASHION64x64_Translated', batch_size=100)
            shift_eq_acc = Metric.cal_acc(encoder=self.encoder, quant_conv=self.quant_conv, quantize=self.quantize, device=self.device)
            print(f'\nTrans_eq_acc: {shift_eq_acc}')
            self.log('Trnas_eq_acc', shift_eq_acc)

        x = self.get_input(batch, self.image_key)
        xrec, emb_loss = self(x)
        qloss = emb_loss[0] + emb_loss[1]
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        num_gpus = self.num_gpus
        max_epochs = self.max_epochs
        train_loader = self.train_dataloader()

        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class CodebookDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        path = Path(root)
        self.image_files = [*path.glob('*.png')]
        self.image_files.sort()
        assert len(self.image_files) % 5 == 0, "MUST be  <len(image_files) % 5 == 0>"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        image_path = self.image_files[i]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)
        return image


class ShiftEqMetric():
    def __init__(self, root, batch_size):
        self.batch_size = batch_size
        self.correct = 0
        self.tot = 0
        ds = CodebookDataset(root)
        assert batch_size % 5 == 0, "MUST be  <batch_size % 5 == 0>"
        self.loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)
        print("\nnumber of shift-eq test image files:", len(ds))

    
    def cal_acc(self, encoder, quant_conv, quantize, device):
        print("encoder is training mode:", encoder.training)
        num_downsampling = encoder.num_resolutions - 1
        with torch.no_grad():
            for image in tqdm(self.loader):
                image = image.to(device)
                B, ch, H, W = image.shape
                fmap_H = int(H/(2**num_downsampling))
                fmap_W = int(W/(2**num_downsampling))
                h = encoder(image)
                h = quant_conv(h)
                quant, emb_loss, info = quantize(h)
                quantized_idx = info[2]
                quantized_idx = quantized_idx.view(self.batch_size, fmap_H, fmap_W)
                splits = quantized_idx.split(5, dim=0)
                for s in splits:
                    CEN = s[0, fmap_H//4:-fmap_H//4, fmap_W//4:-fmap_W//4]
                    UL = s[1, :fmap_H//2, :fmap_W//2]
                    UR = s[2, :fmap_H//2, fmap_W//2:]
                    LL = s[3, fmap_H//2:, :fmap_W//2]
                    LR = s[4, fmap_H//2:, fmap_W//2:]
                    combi = combinations([CEN, UL, UR, LL, LR], 2)
                    for A, B in combi:
                        correct = A == B
                        num_correct = correct.sum()
                        self.correct += num_correct.item()
                        self.tot += A.size(0) * A.size(1)
        
        return self.correct / self.tot
