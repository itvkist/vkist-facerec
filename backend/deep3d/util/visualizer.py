"""This script defines the visualizer for Deep3DFaceRecon_pytorch
"""

import numpy as np
import os
import time
from . import util

# NOTE: html (dominate) and Visualizer are only needed for training.
# They are commented out to remove the dominate dependency at inference time.

# from . import util, html
# from subprocess import Popen, PIPE
# from torch.utils.tensorboard import SummaryWriter

# def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
#     """Save images to the disk. Training only."""
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#     webpage.add_header(name)
#     ims, txts, links = [], [], []
#     for label, im_data in visuals.items():
#         im = util.tensor2im(im_data)
#         image_name = '%s/%s.png' % (label, name)
#         os.makedirs(os.path.join(image_dir, label), exist_ok=True)
#         save_path = os.path.join(image_dir, image_name)
#         util.save_image(im, save_path, aspect_ratio=aspect_ratio)
#         ims.append(image_name)
#         txts.append(label)
#         links.append(image_name)
#     webpage.add_images(ims, txts, links, width=width)


# class Visualizer():
#     """Training-only visualizer using tensorboard and dominate HTML."""
#     ... (removed — not used at inference)


class MyVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')

        # SummaryWriter and log file are only needed during training
        if opt.phase != 'test':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, save_path, visuals, total_iters, epoch, dataset='train',
                                 save_results=False, count=0, name=None, add_image=True):
        """Display/save results to tensorboard. Training only."""
        for label, image in visuals.items():
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d' % (dataset, i + count),
                                          image_numpy, total_iters, dataformats='HWC')
                if save_results:
                    img_path = os.path.join(save_path, '%s.png' % name) if name is not None \
                        else os.path.join(save_path, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)

    def plot_current_losses(self, total_iters, losses, dataset='train'):
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s' % dataset, value, total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, dataset='train'):
        message = '(dataset: %s, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            dataset, epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_img(self, visuals):
        """Return the rendered image numpy array. Used at inference."""
        for label, image in visuals.items():
            for i in range(image.shape[0]):
                return util.tensor2im(image[i])
