# import sys
# import random
import time
import pygame
import numpy as np
from ..systems.simulation import Simulation


class PygameShapeDrawer:
    def __init__(self, world_area_side_len, cells_number_vertical, cells_number_horizontal, shape_string = "circle"):
        self.cells_number_vertical = cells_number_vertical
        self.cells_number_horizontal = cells_number_horizontal
        self.shape_string = shape_string
        if self.cells_number_vertical != self.cells_number_horizontal:
            self.shape_string = "rectangle"
        self.w = world_area_side_len / self.cells_number_horizontal
        self.h = world_area_side_len / self.cells_number_vertical
        self.diameter = min(self.w, self.h) # bit area side size / square area side size
        self.radius = self.diameter / 2

    def draw_circle(self, screen, color, x, y):
        return pygame.draw.circle(screen,
                                  color,
                                  [x, y],
                                  self.radius,
                                  0)

    def draw_rectangle(self, screen, color, x, y):
        return pygame.draw.rect(screen, 
                                color, 
                                pygame.Rect(x, y, self.w, self.h))

    def draw_shape(self, screen, color, x, y):
        if self.shape_string == "circle":
            return self.draw_circle(screen, color, x, y)
        elif self.shape_string == "rectangle":
            return self.draw_rectangle(screen, color, x, y)
        else:
            assert False, "not existing shape"
            

class PygameSession:
    def __init__(self, simulation: Simulation, by_mouse: bool, rendering_duration: float = 0.1):
        self.by_mouse = by_mouse
        self.visualization_title = 'Row of Binary Agents'
        # simulation
        self.simulation = simulation 
        self.__simulation_steps = self.simulation.steps_number
        self.bits_number, self.agents_number = self.simulation.group.state.shape
        # pygame
        self.rendering_duration = rendering_duration
        self.screen_size = [1100, 900]
        self.meridian = self.screen_size[0] / 2 - 0.1 * self.screen_size[0]
        self.equator = self.screen_size[1] / 2
        self.screen_color = [250, 250, 250]
        self.bit_color = pygame.Color("aquamarine")
        self.world_area_side_len = 800 
        self.cells_drawer = PygameShapeDrawer(world_area_side_len = self.world_area_side_len, 
                                              cells_number_vertical = self.bits_number, 
                                              cells_number_horizontal = self.agents_number,
                                              shape_string = "circle")
        # writings
        self.font_size = 20

    def simulation_steps(self):
        return self.__simulation_steps

    def metrics_strings(self):
        return [f"Simulation Steps Left: {self.simulation_steps()}",
                f"Number of Agents: {self.agents_number}",
                f"Number of each Agent's bits: {self.bits_number}",
                f"{[int(self.simulation.group.state[i , :].sum()) for i in range(self.bits_number)]}",
                ] # + [f"{self.simulation.group.adjacency_matrix[j]}" for j in range(self.agents_number)]
        
    def draw_text(self, text: str, x, y):
        green = (0, 255, 0)
        blue = (0, 0, 128)
        # create a text surface object,
        # on which text is drawn on it.
        text = self.font.render(text, True, green, blue)
        # create a rectangular object for the
        # text surface object
        textRect = text.get_rect()
        # set the center of the rectangular object.
        textRect.center = (x, y)
        # copying the text surface object
        # to the display surface object
        # at the center coordinate.
        self.screen.blit(text, textRect)
        
    def draw_metrics_table(self):
        table_coords = [0.85 * self.screen_size[0], 0.03 * self.screen_size[1]]
        self.font = pygame.font.Font('freesansbold.ttf', self.font_size)
        metrics_strings = self.metrics_strings()
        for m in range(len(metrics_strings)):
            metric_string = metrics_strings[m]
            self.draw_text(metric_string, table_coords[0], table_coords[1] + m * self.font_size)

    def draw_bit(self, bit: bool, x, y):
        return self.cells_drawer.draw_shape(self.screen,
                                            self.bit_color if bit else self.screen_color,
                                            x,
                                            y)
        
    def draw_agent_state(self, agent_state, x, y):
        for i in range(agent_state.shape[0]):
            bit_value = agent_state[i]
            assert bit_value in [0., 1.], f"some bits are not binary. {self.simulation.group.state=}"
            bit = self.draw_bit((bit_value == 1), x, y + i * self.cells_drawer.h) 
    
    def draw_group_state(self, group_state: np.ndarray):
        x_for_all = self.meridian - self.cells_drawer.w * (self.agents_number - 1) / 2
        y_for_all = self.equator - self.cells_drawer.h * (self.bits_number - 1) / 2
        for j in range(self.agents_number):
            self.draw_agent_state(group_state[: , j].reshape((self.bits_number, 1)), x_for_all + (j) * self.cells_drawer.w, y_for_all)

    def draw_snap(self, group_state: np.ndarray):
        self.draw_group_state(group_state)
        pygame.display.flip()
        self.__simulation_steps -= 1
        self.continue_simulation = False
        
    def draw_simulation(self):
        # init simulation screen
        pygame.init()
        pygame.display.set_caption(self.visualization_title)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill(self.screen_color)
        # draw init state
        self.draw_snap(self.simulation.group.state)
        # running rutin
        self.running = True
        while self.running:
            time.sleep(self.rendering_duration)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.continue_simulation = True
            if not self.by_mouse:
                self.continue_simulation = True
            if self.continue_simulation and self.__simulation_steps > 0:
                self.draw_snap(next(self.simulation))
            self.draw_metrics_table()
        pygame.quit()        

    


