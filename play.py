import gym
import pygame
from gym.utils.play import play

mapping = {
    (pygame.K_LSHIFT, pygame.K_LEFT): 1,
    (pygame.K_LSHIFT, pygame.K_RIGHT): 2,
    (pygame.K_LSHIFT, ord(" ")): 3,
}
play(
    gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False),
    keys_to_action=mapping,
)
