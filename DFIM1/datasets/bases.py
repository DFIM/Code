from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from PIL import ImageFilter
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
import numpy as np

def colorful_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * np.sqrt(ratio))
    w_crop = int(w * np.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img2_abs_[h_start:h_start + h_crop,
                                                                         w_start:w_start + w_crop] + (
                                                                           1 - lam) * img1_abs_[
                                                                                      h_start:h_start + h_crop,
                                                                                      w_start:w_start + w_crop]
    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))

    img21 = Image.fromarray(img21.astype(np.uint8))

    return img21

def get_pre_transform():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(0.5),
        T.Pad(10),
        T.RandomCrop((224, 224)),

    ])
    return transform

def get_aft_transform():
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(scale=(0.02, 0.4), value=mean),
    ])
    return transform

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("TIRRS.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
            self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
            self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

def get_self_supervised_augmentation(img_size):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.), antialias=True),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return aug

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 aug_ss: bool = True
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.augmentation_ss = get_self_supervised_augmentation((224, 224))
        self.aug_ss = aug_ss

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        #img = read_image(img_path)
        image_pil = Image.open(img_path)

        if self.aug_ss:
            aug_ss_1 = self.augmentation_ss(image_pil)
            aug_ss_2 = self.augmentation_ss(image_pil)

        if self.transform is not None:
            img = self.transform(image_pil.convert('RGB'))

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        if self.aug_ss:
            ret = {
                'index': index,
                'pids': pid,
                'image_ids': image_id,
                'images': img,
                'aug_ss_1': aug_ss_1,
                'aug_ss_2': aug_ss_2,
                'caption_ids': tokens,
            }
        else:
            ret = {
                'index': index,
                'pids': pid,
                'image_ids': image_id,
                'images': img,
                'caption_ids': tokens,
            }

        return ret




class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 select_strategy='different_pid'):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()
        self.select_strategy = select_strategy
        self.get_pre_transform = get_pre_transform()
        self.get_aft_transform = get_aft_transform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)

        img1 = self.get_pre_transform(img)
        selected_img = self.select_image(pid, index)
        img2 = self.get_pre_transform(selected_img)
        aug_img1 = colorful_spectrum_mix(np.asarray(img1), np.asarray(img2))
        aug_img1 = self.get_aft_transform(aug_img1)

        if self.transform is not None:
            images = self.transform(img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': images,
            'aug_images': aug_img1,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
        }

        return ret

    def select_image(self, pid, current_index):
        if self.select_strategy == 'same_pid':
            selected_index = self._select_same_pid(pid, current_index)
        elif self.select_strategy == 'different_pid':
            selected_index = self._select_different_pid(pid, current_index)
        else:  # random
            selected_index = self._select_random(current_index)

        selected_img_path = self.dataset[selected_index][2]
        selected_img = read_image(selected_img_path)

        return selected_img

    def _select_same_pid(self, pid, current_index):
        same_pid_indices = [i for i, data in enumerate(self.dataset) if data[0] == pid and i != current_index]
        if same_pid_indices:
            return random.choice(same_pid_indices)
        else:
            return current_index  # 如果没有其他同 pid 图像，返回当前图像

    def _select_different_pid(self, pid, current_index):
        different_pid_indices = [i for i, data in enumerate(self.dataset) if data[0] != pid]
        if different_pid_indices:
            return random.choice(different_pid_indices)
        else:
            return current_index  # 如果没有不同 pid 图像，返回当前图像

    def _select_random(self, current_index):
        all_indices = list(range(len(self.dataset)))
        all_indices.remove(current_index)
        return random.choice(all_indices)

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

# class ImageTextMLMDataset(Dataset):
#     def __init__(self,
#                  dataset,
#                  transform=None,
#                  text_length: int = 77,
#                  truncate: bool = True,
#                  aug_ss: bool = True
#                  ):
#         self.dataset = dataset
#         self.transform = transform
#         self.text_length = text_length
#         self.truncate = truncate
#         self.augmentation_ss = get_self_supervised_augmentation((256, 256))
#
#         self.tokenizer = SimpleTokenizer()
#         self.aug_ss = aug_ss
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         pid, image_id, img_path, caption = self.dataset[index]
#
#         #img = read_image(img_path)
#         image_pil = Image.open(img_path)
#
#         if self.aug_ss:
#             aug_ss_1 = self.augmentation_ss(image_pil)
#             aug_ss_2 = self.augmentation_ss(image_pil)
#
#         if self.transform is not None:
#             img = self.transform(image_pil.convert('RGB'))
#
#         caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
#                                   truncate=self.truncate)
#
#         mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
#
#         if self.aug_ss:
#             ret = {
#                 'pids': pid,
#                 'image_ids': image_id,
#                 'images': img,
#                 'aug_ss_1': aug_ss_1,
#                 'aug_ss_2': aug_ss_2,
#                 'caption_ids': caption_tokens,
#                 'mlm_ids': mlm_tokens,
#                 'mlm_labels': mlm_labels
#             }
#         else:
#             ret = {
#                 'pids': pid,
#                 'image_ids': image_id,
#                 'images': img,
#                 'caption_ids': caption_tokens,
#                 'mlm_ids': mlm_tokens,
#                 'mlm_labels': mlm_labels
#             }
#
#         return ret
#
#
#     def _build_random_masked_tokens_and_labels(self, tokens):
#         """
#         Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
#         :param tokens: list of int, tokenized sentence.
#         :return: (list of int, list of int), masked tokens and related labels for MLM prediction
#         """
#         mask = self.tokenizer.encoder["<|mask|>"]
#         token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405
#
#         labels = []
#         for i, token in enumerate(tokens):
#             if 0 < token < 49405:
#                 prob = random.random()
#                 # mask token with 15% probability
#                 if prob < 0.15:
#                     prob /= 0.15
#
#                     # 80% randomly change token to mask token
#                     if prob < 0.8:
#                         tokens[i] = mask
#
#                     # 10% randomly change token to random token
#                     elif prob < 0.9:
#                         tokens[i] = random.choice(token_range)
#
#                     # -> rest 10% randomly keep current token
#
#                     # append current token to output (we will predict these later)
#                     labels.append(token)
#                 else:
#                     # no masking token (will be ignored by loss function later)
#                     labels.append(0)
#             else:
#                 labels.append(0)
#
#         if all(l == 0 for l in labels):
#             # at least mask 1
#             labels[1] = tokens[1]
#             tokens[1] = mask
#
#         return torch.tensor(tokens), torch.tensor(labels)