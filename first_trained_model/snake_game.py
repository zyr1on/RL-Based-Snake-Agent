import random, collections
import numpy as np
import pygame

GRID = 20
CELL = 30
WIN  = GRID * CELL
FPS  = 30

class SnakeEnv:
    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # yukarı, sağ, aşağı, sol

    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = self.clock = self.font = None
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN, WIN))
            pygame.display.set_caption("Snake RL")
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("monospace", 22)
        self.reset()

    def reset(self):
        cx, cy = GRID // 2, GRID // 2
        self.snake     = collections.deque([(cx,cy),(cx-1,cy),(cx-2,cy)])
        self.dir_i     = 1
        self.score     = 0
        self.steps     = 0
        self.max_steps = GRID * GRID * 2
        self._place_food()
        return self._obs()

    def _place_food(self):
        empty = [(x,y) for x in range(GRID) for y in range(GRID)
                 if (x,y) not in self.snake]
        self.food = random.choice(empty) if empty else (0,0)

    # aksiyon: 0=düz, 1=sağa, 2=sola
    def step(self, action):
        self.steps += 1
        self.dir_i = (self.dir_i + (1 if action==1 else -1 if action==2 else 0)) % 4
        dx, dy = self.DIRS[self.dir_i]
        hx, hy = self.snake[0]
        nx, ny = hx+dx, hy+dy

        if nx<0 or nx>=GRID or ny<0 or ny>=GRID or (nx,ny) in self.snake:
            return self._obs(), -10.0, True

        self.snake.appendleft((nx,ny))
        if (nx,ny) == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.01

        return self._obs(), reward, self.steps >= self.max_steps

    def _obs(self):
        hx, hy = self.snake[0]
        dx, dy = self.DIRS[self.dir_i]
        lx, ly =  dy, -dx
        rx, ry = -dy,  dx

        def danger(ox, oy):
            nx, ny = hx+ox, hy+oy
            return float(nx<0 or nx>=GRID or ny<0 or ny>=GRID or (nx,ny) in self.snake)

        fx, fy = self.food
        return np.array([
            danger(dx,dy), danger(lx,ly), danger(rx,ry),
            float(dx== 1), float(dx==-1),
            float(dy==-1), float(dy== 1),
            float(fx>hx),  float(fx<hx),
            float(fy<hy),  float(fy>hy),
        ], dtype=np.float32)

    def render(self):
        if not self.render_mode:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close(); exit()

        self.screen.fill((15,15,15))
        for i,(x,y) in enumerate(self.snake):
            c = (50,200,50) if i==0 else (30,150,30)
            pygame.draw.rect(self.screen, c,
                             (x*CELL+1, y*CELL+1, CELL-2, CELL-2), border_radius=4)
        fx, fy = self.food
        pygame.draw.ellipse(self.screen, (220,50,50),
                            (fx*CELL+3, fy*CELL+3, CELL-6, CELL-6))
        txt = self.font.render(f"Score: {self.score}", True, (220,220,220))
        self.screen.blit(txt, (6,4))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()