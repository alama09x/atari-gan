# Generative Adversarial Network for Atari Images
# Credit: "Deep Reinforcement Learning Hands-On" by Maxim Lapan

import cv2
import time
import random
import argparse
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.utils as vutils

import gymnasium as gym
from gymnasium import spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing wrapper for input images.

    Observation space is a Gym Box, meaning a multidimensional
    array with a fixed range for each dimension
    
    In this case: the unwrapped observation space is
    `Box(low=0, high=255, shape=(210, 160, 3))`,
    
    While the wrapper reshapes this to:
    `Box(low=0, high=255, shape=(3, 64, 64))`.
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)
    
    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        # Resize the image from 210*160 to IMAGE_SIZE*IMAGE_SIZE
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # Move color dimension to beginning
        new_obs = np.moveaxis(new_obs, 2, 0)

        return new_obs.astype(np.float32)

class Discriminator(nn.Module):
    """
    The Discriminator in a General Adversarial Network learns to
    distinguish between real Atari images and fake ones created by the
    Generator.
    """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*2, out_channels=DISCR_FILTERS*4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*4, out_channels=DISCR_FILTERS*8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        # TODO: find out what this means
        return conv_out.view(-1, 1).squeeze(dim=1)

class Generator(nn.Module):
    """
    The Generator in a General Adversarial Network learns to generate
    fake images to fool the Discriminator.
    """
    def __init__(self, output_shape):
        super(Generator, self).__init__()
    
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE,
                               out_channels=GENER_FILTERS*8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS*8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS*8,
                               out_channels=GENER_FILTERS*4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS*4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS*4,
                               out_channels=GENER_FILTERS*2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS*2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS*2,
                               out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS,
                               out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.pipe(x)

def iterate_batches(envs: tt.List[gym.Env],
                    batch_size: int = BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
    # Store observations in periodically cleared batch,
    # ensuring that the environment is reset before use
    batch = [e.reset()[0] for e in envs]

    # Sample the environment
    env_gen = iter(lambda: random.choice(envs), None)
    
    while True:
        e = next(env_gen)
        action = e.action_space.sample()

        # Step through one action
        obs, reward, is_done, is_trunc, _ = e.step(action)
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Return batch
            batch_np = np.array(batch, dtype=np.float32)
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)
            batch.clear()
        if is_done or is_trunc:
            e.reset()

if __name__ == "__main__":
    # Specify the device to run on as a command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()
    
    device = torch.device(args.dev)
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')
    ]
    shape = envs[0].observation_space.shape

    # Initialize modules to device
    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    # Binary cross-entropy loss
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(),
                               lr=LEARNING_RATE,
                               # TODO: what are betas?
                               betas=(0.5, 0.999))
    
    dis_optimizer = optim.Adam(params=net_discr.parameters(),
                               lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    
    # Tensorboard logging
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []

    # Index
    iter_no = 0
    
    # 1s are true images and 0s are fake
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    ts_start = time.time()

    for batch_v in iterate_batches(envs):
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # Train the discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # Train the generator based on the output of the discriminator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        # Periodic logging
        if iter_no % REPORT_EVERY_ITER == 0: 
            dt = time.time() - ts_start 
            log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e", 
                     iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
            
            ts_start = time.time()

            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
    
        # Show images on Tensorboard
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
            writer.add_image("fake", img, iter_no)
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("real", img, iter_no)