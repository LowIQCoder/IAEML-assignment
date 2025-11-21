from functools import partial
import jax
import pygame
import jax.numpy as jnp
from jax import random
import math

from utils import utils

FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (30, 30, 30)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 200, 50)
YELLOW = (200, 200, 50)

# Car
CAR_LENGTH = 80
CAR_WIDTH = 40
WHEELBASE = 60
STERING_RATE = math.radians(10)
ACCELERATION = 0.2

# TODO: draw time

class PygameFrontend:
    def __init__(self, env, env_params, init_state, eval_mode=False, agent_fn=None):
        pygame.init()
        self.env = env
        self.params = env_params
        self.init_state = init_state
        self.eval_mode = eval_mode
        self.agent_fn = agent_fn

        h, w = int(env_params.map_height_width[0]), int(env_params.map_height_width[1])
        self.CELL_SIZE = 80  # pixels per map unit
        self.screen = pygame.display.set_mode((w * self.CELL_SIZE, h * self.CELL_SIZE))
        pygame.display.set_caption("2D Env Renderer")

        # Correct collider radius in pixels
        self.COLLIDER_RADIUS_PX = self.CELL_SIZE // 2

        # Agent and goal match collider size
        self.AGENT_RADIUS_PX = self.COLLIDER_RADIUS_PX
        self.GOAL_RADIUS_PX = self.COLLIDER_RADIUS_PX
        # Obstacle square fits one grid cell
        self.OBSTACLE_SIDE_PX = self.CELL_SIZE

        self.clock = pygame.time.Clock()
        self.key = random.PRNGKey(0)
        self.obs, self.state = self.env.reset(self.key, env_params, init_state)
        self.running = True
        rotate_to_map_f = partial(
            utils.convert_to_map_view, map_shape=env_params.map_height_width
        )
        self.rotate_to_map_vmap = jax.vmap(rotate_to_map_f, in_axes=(0))

    def draw_square_with_striped_circle(
        self, center_px, square_side_px, circle_radius_px, square_color, num_stripes=6
    ):
        # Draw filled square (one cell)
        pygame.draw.rect(
            self.screen,
            square_color,
            (
                center_px[0] - square_side_px // 2,
                center_px[1] - square_side_px // 2,
                square_side_px,
                square_side_px,
            ),
        )
        # Draw stripes inside circle
        for i in range(-num_stripes, num_stripes + 1):
            y_offset = i * circle_radius_px / num_stripes
            half_width = math.sqrt(max(circle_radius_px**2 - y_offset**2, 0))
            start_pos = (int(center_px[0] - half_width), int(center_px[1] + y_offset))
            end_pos = (int(center_px[0] + half_width), int(center_px[1] + y_offset))
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)

    def draw_path(self, path_array: jnp.ndarray, agent_pos_map: jnp.ndarray):
        """
        Draws a path over the map in blue.
        - path_array: [N, 2] deltas in map coordinates (h, w)
        - agent_pos_map: starting position of agent in map coordinates (h, w)
        """
        if path_array.shape[0] == 0:
            return  # nothing to draw

        waypoints = jnp.vstack([agent_pos_map, path_array])

        # Convert map positions to pixel coordinates
        pixels = [
            (int((wp[1] + 0.5) * self.CELL_SIZE), int((wp[0] + 0.5) * self.CELL_SIZE))
            for wp in waypoints
        ]

        # Draw lines connecting waypoints
        for start, end in zip(pixels[:-1], pixels[1:]):
            pygame.draw.line(self.screen, BLUE, start, end, 3)
            
    def draw_rays_transparent(self, screen: pygame.Surface, agent_pos, rays, cell_size, alpha=80):
        """
        Draw transparent red rays from the agent's position.

        Parameters:
        - screen: pygame.Surface
        - agent_pos: (x, y) in map/world coordinates
        - rays: (N, 2) array of vectors (magnitude = distance)
        - cell_size: float, scale factor from map units to pixels
        - alpha: transparency (0=fully transparent, 255=opaque)
        """
        # Ensure we have a valid surface
        screen_width, screen_height = screen.get_size()
        ray_surf = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        agent_px = (int((agent_pos[1] + 0.5) * cell_size),
                    int((agent_pos[0] + 0.5) * cell_size))

        for vec in rays:
            end_px = (int(agent_px[0] + vec[1] * cell_size),
                    int(agent_px[1] + vec[0] * cell_size))
            pygame.draw.line(ray_surf, (255, 0, 0, alpha), agent_px, end_px, 2)

        screen.blit(ray_surf, (0, 0))

    def draw_info(self, screen: pygame.Surface, info: dict, position=(10, 10), font_size=15, line_height=25, color=(255, 255, 255)):
        """
        Draws a dictionary of info on the screen.

        Parameters:
        - screen: pygame.Surface
        - info: dict, e.g., {"time": 10, "distance_to_path": 5.2}
        - position: (x, y) top-left corner for text
        - font_size: int, font size
        - line_height: vertical spacing between lines
        - color: RGB tuple
        """
        font = pygame.font.SysFont("Arial", font_size)
        x, y = position

        for key, value in info.items():
            if key == "direction_of_path":
                value = f"(x={value[0]:.2f}, y={value[1]:.2f})"
            if key == "distance_to_path":
                value = f"{value:.4f}"
            text_surf = font.render(f"{key}: {value}", True, color)
            screen.blit(text_surf, (x, y))
            y += line_height

    def draw_agent(self, agent_pos, theta, phi):
        # --- Compute axle positions ---
        y, x = (int((agent_pos[0] + 0.7) * self.CELL_SIZE), int((agent_pos[1] + 0.5) * self.CELL_SIZE))
        rear_axle = (x, y)
        front_axle = (
            x + WHEELBASE * math.cos(theta),
            y + WHEELBASE * math.sin(theta)
        )

        # Draw rear axle (black line)
        rear_left = (
            rear_axle[0] - (CAR_WIDTH / 2) * math.sin(theta),
            rear_axle[1] + (CAR_WIDTH / 2) * math.cos(theta)
        )
        rear_right = (
            rear_axle[0] + (CAR_WIDTH / 2) * math.sin(theta),
            rear_axle[1] - (CAR_WIDTH / 2) * math.cos(theta)
        )
        pygame.draw.line(self.screen, (0, 0, 0), rear_left, rear_right, 4)

        # Draw front axle (red line, steered by phi)
        front_left = (
            front_axle[0] - (CAR_WIDTH / 2) * math.sin(theta + phi),
            front_axle[1] + (CAR_WIDTH / 2) * math.cos(theta + phi)
        )
        front_right = (
            front_axle[0] + (CAR_WIDTH / 2) * math.sin(theta + phi),
            front_axle[1] - (CAR_WIDTH / 2) * math.cos(theta + phi)
        )
        pygame.draw.line(self.screen, (200, 0, 0), front_left, front_right, 4)

        # Draw body (connecting axles)
        pygame.draw.line(self.screen, (50, 50, 200), rear_axle, front_axle, 3)

        # Draw wheel center points
        pygame.draw.circle(self.screen, (0, 0, 0), (int(rear_axle[0]), int(rear_axle[1])), 4)
        pygame.draw.circle(self.screen, (200, 0, 0), (int(front_axle[0]), int(front_axle[1])), 4)
    

    def draw(self):
        self.screen.fill(WHITE)

        # To map coordinates
        path_array = self.rotate_to_map_vmap(self.state.path_array)
        static_obst_map = self.rotate_to_map_vmap(self.state.static_obstacles)
        kin_obst_map = self.rotate_to_map_vmap(self.state.kinematic_obstacles)

        agent_pos = self.rotate_to_map_vmap(jnp.atleast_2d(self.state.agent_pos))[0]
        goal_pos = self.rotate_to_map_vmap(jnp.atleast_2d(self.state.goal_pos))[0]
        rays = jnp.flip(self.obs.collision_rays, axis=-1) * jnp.array([-1, 1])[None, :]

        # draw path
        self.draw_path(path_array, agent_pos)

        # Draw kinematic obstacles
        for pos in kin_obst_map:
            y, x = pos
            center_px = (
                int((x + 0.5) * self.CELL_SIZE),
                int((y + 0.5) * self.CELL_SIZE),
            )
            self.draw_square_with_striped_circle(
                center_px, self.OBSTACLE_SIDE_PX, self.COLLIDER_RADIUS_PX, RED
            )

        # Draw static obstacles
        for pos in static_obst_map:
            y, x = pos
            center_px = (
                int((x + 0.5) * self.CELL_SIZE),
                int((y + 0.5) * self.CELL_SIZE),
            )
            self.draw_square_with_striped_circle(
                center_px, self.OBSTACLE_SIDE_PX, self.COLLIDER_RADIUS_PX, DARK_GRAY
            )

        # Draw goal
        y, x = goal_pos
        center_px = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
        pygame.draw.circle(self.screen, GREEN, center_px, self.GOAL_RADIUS_PX)

        # draw rays after obstacles
        self.draw_rays_transparent(self.screen, agent_pos, rays, self.CELL_SIZE)
        
        # Draw agent      
        self.draw_agent(agent_pos, -self.state.theta, self.state.phi)
        
        # Draw info
        self.draw_info(self.screen, self.info) 

        pygame.display.flip()

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        dv = 0
        dphi = 0
        if keys[pygame.K_UP]:
            dv = ACCELERATION
        if keys[pygame.K_DOWN]:
            dv = -ACCELERATION
        if keys[pygame.K_LEFT]:
            dphi = -STERING_RATE
        if keys[pygame.K_RIGHT]:
            dphi = STERING_RATE

        action = jnp.array([dv, dphi], dtype=jnp.float32)
        return action

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.eval_mode and self.agent_fn is not None:
                action = self.agent_fn(self.state)
            else:
                action = self.handle_keys()

            self.obs, self.state, _, _, self.info = self.env.step(
                self.key, self.state, action, self.params
            )

            # print(self.state.static_obstacles)
            # print(self.obs.collision_rays)

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
