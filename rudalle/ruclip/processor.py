# -*- coding: utf-8 -*-
import os
import json
import torch
import youtokentome as yttm
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence


class RuCLIPProcessor:
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(self, tokenizer_path, image_size=224, text_seq_length=76, mean=None, std=None):

        self.tokenizer = yttm.BPE(tokenizer_path)
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        self.text_seq_length = text_seq_length
        self.image_size = image_size

    def encode_text(self, text):
        text = text.lower()
        tokens = self.tokenizer.encode([text], output_type=yttm.OutputType.ID, dropout_prob=0.0)[0]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        tokens = tokens[:self.text_seq_length]
        mask = [1] * len(tokens)
        return torch.tensor(tokens).long(), torch.tensor(mask).long()

    def decode_text(self, encoded):
        return self.tokenizer.decode(encoded.cpu().numpy().tolist(), ignore_ids=[
            self.eos_id, self.bos_id, self.unk_id, self.pad_id
        ])[0]

    def __call__(self, text=None, images=None, **kwargs):
        inputs = {}
        if text is not None:
            input_ids, masks = [], []
            texts = [text] if isinstance(text, str) else text
            for text in texts:
                tokens, mask = self.encode_text(text)
                input_ids.append(tokens)
                masks.append(mask)
            inputs['input_ids'] = pad_sequence(input_ids, batch_first=True)
            inputs['attention_mask'] = pad_sequence(masks, batch_first=True)
        if images is not None:
            pixel_values = []
            for i, image in enumerate(images):
                pixel_values.append(self.image_transform(image))
            inputs['pixel_values'] = pad_sequence(pixel_values, batch_first=True)
        return inputs

    @classmethod
    def from_pretrained(cls, folder):
        tokenizer_path = os.path.join(folder, 'bpe.model')
        config = json.load(open(os.path.join(folder, 'config.json')))
        image_size = config['vision_config']['image_size']
        text_seq_length = config['text_config']['max_position_embeddings'] - 1
        mean, std = config.get('mean'), config.get('std')
        return cls(tokenizer_path, image_size=image_size, text_seq_length=text_seq_length, mean=mean, std=std)
