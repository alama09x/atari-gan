import cv2
import time
import random
import argparse
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Engine, Events 
from ignite.handlers import Timer 
from ignite.metrics import RunningAverage 
from ignite.contrib.handlers import tensorboard_logger as tb_logger

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
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)
    
    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

class Discriminator(nn.Module):
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
        return conv_out.view(-1, 1).squeeze(dim=1)

class Generator(nn.Module):
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

def process_batch(trainer, batch):
    gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
    gen_input_v.normal_(0, 1)
    gen_input_v = gen_input_v.to(device)
    batch_v = batch.to(device)
    gen_output_v = net_gener(gen_input_v)

    dis_optimizer.zero_grad()
    dis_output_true_v = net_discr(batch_v)
    dis_output_fake_v = net_discr(gen_output_v.detach())
    dis_loss = objective(dis_output_true_v, true_labels_v) + \
               objective(dis_output_fake_v, fake_labels_v)
    dis_loss.backward()

    gen_optimizer.zero_grad()
    dis_output_v = net_discr(gen_output_v)
    gen_loss = objective(dis_output_v, true_labels_v)
    gen_loss.backward()
    gen_optimizer.step()

    if trainer.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
        img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
        trainer.tb.writer.add_image("fake", img, iter_no)
        img = vutils.make_grid(batch_v.data[:64], normalize=True)
        trainer.tb.writer.add_image("real", img, iter_no)
        trainer.tb.writer.flush()
    return dis_loss.item(), gen_loss.item()

def iterate_batches(envs: tt.List[gym.Env],
                    batch_size: int = BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)
    
    while True:
        e = next(env_gen)
        action = e.action_space.sample()
        obs, reward, is_done, is_trunc, _ = e.step(action)
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)
            batch.clear()
        if is_done or is_trunc:
            e.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()
    
    device = torch.device(args.dev)
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')
    ]
    shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(),
                               lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    
    dis_optimizer = optim.Adam(params=net_discr.parameters(),
                               lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0
    
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    engine = Engine(process_batch)
    tb = tb_logger.TensorboardLogger(log_dir=None)
    engine.tb = tb
    RunningAverage(output_transform=lambda out: out[1]).\
        attach(engine, "avg_loss_gen")
    RunningAverage(output_transform=lambda out: out[0]).\
        attach(engine, "avg_loss_dis")
    
    handler = tb_logger.OutputHandler(tag="train", metric_names=[
        'avg_loss_gen', 'avg_loss_dis'])
    
    tb.attach(engine, log_handler=handler, event_name=Events.ITERATION_COMPLETED)
    timer = Timer()
    timer.attach(engine)

    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(trainer):
        if trainer.state.iteration % REPORT_EVERY_ITER == 0:
            log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e", 
                     trainer.state.iteration, timer.value(),
                     trainer.state.metrics['avg_loss_gen'],
                     trainer.state.metrics['avg_loss_dis'])
            timer.reset()

    engine.run(data=iterate_batches(envs))