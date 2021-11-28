# -*- coding: utf-8 -*-
import os
from glob import glob
from os.path import join

import cv2
import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image

from . import utils


def generate_images(text, tokenizer, dalle, vae, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                    seed=None, use_cache=True):
    # TODO docstring
    if seed is not None:
        utils.seed_everything(seed)

    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')

    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    pil_images, scores = [], []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            has_cache = False
            sample_scores = []
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(out.shape[1], total_seq_length)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    logits, has_cache = dalle(out, attention_mask,
                                              has_cache=has_cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    sample_scores.append(probs[torch.arange(probs.size(0)), sample.transpose(0, 1)])
                    out = torch.cat((out, sample), dim=-1)
            codebooks = out[:, -image_seq_length:]
            images = vae.decode(codebooks)
            pil_images += utils.torch_tensors_to_pil_list(images)
            scores += torch.cat(sample_scores).sum(0).detach().cpu().numpy().tolist()
    return pil_images, scores


def super_resolution(pil_images, realesrgan, batch_size=4):
    result = []
    for pil_image in pil_images:
        with torch.no_grad():
            sr_image = realesrgan.predict(np.array(pil_image), batch_size=batch_size)
        result.append(sr_image)
    return result


def cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device='cpu', count=4):
    with torch.no_grad():
        inputs = ruclip_processor(text=text, images=pil_images)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        outputs = ruclip(**inputs)
        sims = outputs.logits_per_image.view(-1).softmax(dim=0)
        items = []
        for index, sim in enumerate(sims.cpu().numpy()):
            items.append({'img_index': index, 'cosine': sim})
    items = sorted(items, key=lambda x: x['cosine'], reverse=True)[:count]
    top_pil_images = [pil_images[x['img_index']] for x in items]
    top_scores = [x['cosine'] for x in items]
    return top_pil_images, top_scores


def show(pil_images, nrow=4, size=14, save_dir=None, show=True):
    """
    :param pil_images: list of images in PIL
    :param nrow: number of rows
    :param size: size of the images
    :param save_dir: dir for separately saving of images, example: save_dir='./pics'
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        count = len(glob(join(save_dir, 'img_*.png')))
        for i, pil_image in enumerate(pil_images):
            pil_image.save(join(save_dir, f'img_{count+i}.png'))

    imgs = torchvision.utils.make_grid(utils.pil_list_to_torch_tensors(pil_images), nrow=nrow)
    if not isinstance(imgs, list):
        imgs = [imgs.cpu()]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        if save_dir is not None:
            count = len(glob(join(save_dir, 'group_*.png')))
            img.save(join(save_dir, f'group_{count+i}.png'))
        if show:
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if show:
        fix.show()
        plt.show()


def convert_emoji_to_rgba(pil_images, emojich_unet,  device='cpu', bs=4):
    final_images = []
    with torch.no_grad():
        for chunk in more_itertools.chunked(pil_images, bs):
            images = []
            for pil_image in chunk:
                image = np.array(pil_image.resize((512, 512)))[:, :, :3]
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
                images.append(image)
            images = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
            pred_masks = emojich_unet(images.to(device))[:, 0, :, :]
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).int().cpu().numpy()
            pred_masks = (pred_masks * 255).astype(np.uint8)
            for pil_image, pred_mask in zip(chunk, pred_masks):
                ret, thresh = cv2.threshold(pred_mask, 0, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(pred_mask, contours, -1, (0, 0, 0), 1)
                final_image = np.zeros((512, 512, 4), np.uint8)
                final_image[:, :, :3] = np.array(pil_image.resize((512, 512)))[:, :, :3]
                final_image[:, :, -1] = pred_mask
                final_image = Image.fromarray(final_image)
                final_images.append(final_image)
    return final_images


def show_rgba(rgba_pil_image):
    img = np.array(rgba_pil_image)
    fig, ax = plt.subplots(1, 3, figsize=(10, 10), dpi=100)
    ax[0].imshow(img[:, :, :3])
    ax[1].imshow(img[:, :, -1])
    mask = np.repeat(np.expand_dims(img[:, :, -1] < 128, -1), 3, axis=-1)
    img = img[:, :, :3]
    img[mask[:, :, 0], 0] = 64
    img[mask[:, :, 0], 1] = 255
    img[mask[:, :, 0], 2] = 64
    ax[2].imshow(img)
