import pygame
import numpy as np
import random

class FlappyBirdEnv:
    def __init__(self, width=288, height=512):
        self.width = width
        self.height = height
        self.gravity = 1
        self.jump_strength = -10
        self.pipe_gap = 100
        self.pipe_width = 52
        self.pipe_speed = 3
        self.bird_size = 20
        self.reset()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird RL")

    def reset(self):
        self.bird_y = self.height // 2
        self.bird_vel = 0
        self.score = 0
        self.pipes = [self._create_pipe()]
        self.done = False
        return self._get_state()

    def _create_pipe(self):
        top = random.randint(50, self.height - 150)
        return {'x': self.width, 'top': top}

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Action: 1 = jump
        if action == 1:
            self.bird_vel = self.jump_strength

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

        # Add new pipe if needed
        if self.pipes[-1]['x'] < self.width - 150:
            self.pipes.append(self._create_pipe())

        # Remove passed pipes
        if self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
            self.score += 1

        # Collision detection
        pipe = self.pipes[0]
        if self._collides(pipe):
            self.done = True
            return self._get_state(), -100, True, {}

        # Off-screen (top/bottom)
        if self.bird_y < 0 or self.bird_y > self.height:
            self.done = True
            return self._get_state(), -100, True, {}

        return self._get_state(), 1, False, {}

    def _collides(self, pipe):
        px, pt = pipe['x'], pipe['top']
        py = self.bird_y

        in_x_range = px < 50 < px + self.pipe_width
        in_y_range = not (pt < py < pt + self.pipe_gap)

        return in_x_range and in_y_range

    def _get_state(self):
        pipe = self.pipes[0]
        return np.array([
            self.bird_y / self.height,
            self.bird_vel / 10,
            (pipe['x'] - 50) / self.width,
            pipe['top'] / self.height
        ], dtype=np.float32)

    def render(self):
        self.screen.fill((135, 206, 235))  # Sky blue
        pygame.draw.rect(self.screen, (255, 255, 0), (50, int(self.bird_y), self.bird_size, self.bird_size))

        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34), (pipe['x'], 0, self.pipe_width, pipe['top']))
            pygame.draw.rect(self.screen, (34, 139, 34), (pipe['x'], pipe['top'] + self.pipe_gap, self.pipe_width, self.height))

        pygame.display.update()
        pygame.time.Clock().tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

