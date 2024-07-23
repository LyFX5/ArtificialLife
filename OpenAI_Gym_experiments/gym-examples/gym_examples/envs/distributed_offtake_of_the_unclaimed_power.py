
import copy
from typing import List

import gym
from gym import spaces
import numpy as np
from modules.electrolyser import Electrolyser
from modules.unclaimed_power_generation import PowerSensor, PowerForecaster



class Environment(gym.Env):
    def __init__(self,
                 number_of_devices_in_plant: int = 2,
                 plant_dynamical_model_discretization_interval: float = 5, # secs
                 MDP_discretization_interval: float = 15 * 60, # secs
                 forecast_horizon: float = 6 * 60 * 60, # secs
                 episode_duration: float = 12 * 60 * 60, # secs
                 init_timestamp: int = 1666979019, # secs
                 plan_generation_mode = synthetic
                 ):

        self.number_of_devices_in_plant = number_of_devices_in_plant
        self.plant_dynamical_model_discretization_interval = plant_dynamical_model_discretization_interval
        self.MDP_discretization_interval = MDP_discretization_interval
        self.forecast_horizon = forecast_horizon
        self.unclaimed_power_forecast_length = forecast_horizon // MDP_discretization_interval - 1
        self.number_of_plant_simulation_steps_in_MDP_step = int(MDP_discretization_interval // plant_dynamical_model_discretization_interval)
        self.number_of_MDP_steps_in_episode = int(episode_duration // MDP_discretization_interval)

        self.plant = self.init_plant(plant_dynamical_model_discretization_interval, number_of_devices_in_plant)
        self.plant_state_dim = self.plant.get_state_vector_dimention()

        self.observation_space = spaces.Dict(
            {
                "unclaimed_power_current": spaces.Box(0, number_of_devices_in_plant, shape=(1,), dtype=float),
                "unclaimed_power_forecast": spaces.Box(0, number_of_devices_in_plant, shape=(self.unclaimed_power_forecast_length,), dtype=float),
                #TODO разобраться с границами для plant_state (-np.inf, np.inf)
                "plant_state": spaces.Box(-np.inf, np.inf, shape=(self.plant_state_dim,), dtype=int),
            }
        )

        self._timestamp = init_timestamp
        self._plan_generation_mode = plan_generation_mode


    def init_plant(self, plant_dynamical_model_discretization_interval, number_of_devices_in_plant):
        #TODO
        pass


    def init_power_forecaster(self):
        #TODO
        pass


    def init_power_sensor(self):
        #TODO
        pass


    def _get_obs(self):
        return {
                "plant_state": self._plant_state,
                "unclaimed_power_current": self._unclaimed_power_current_value,
                "unclaimed_power_forecast": self._unclaimed_power_forecast
               }


    def _get_info(self):
        return {
            "total_output_of_plant_at_MDP_step": self._total_output_of_plant_at_MDP_step
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_of_MDP_steps_left = self.number_of_MDP_steps_in_episode

        if not (options is None):
            self._timestamp = options["init_timestamp"]
            self._plan_generation_mode = options["plan_generation_mode"]

        self.plant = self.init_plant(self.number_of_devices_in_plant)
        self.unclaimed_power_sensor = self.init_power_sensor()
        self.unclaimed_power_forecaster = self.init_power_forecaster()

        self._plant_state = self.plant.get_state_vector()
        self._unclaimed_power_current_value = self.unclaimed_power_sensor.get_current_value(self._timestamp, self._plan_generation_mode)
        self._unclaimed_power_forecast = self.unclaimed_power_forecaster.get_forecast(self._timestamp, self._plan_generation_mode)

        self._total_output_of_plant_at_MDP_step = [0] * self.number_of_plant_simulation_steps_in_MDP_step

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action: List[float], options=None):
        U = [0] * len(action)
        for i, a in enumerate(action):
            if a < 0:
                U[i] = 0
            else:
                U[i] = 0.4 * a + 0.6
        if not (options is None):
            if options["made_up_action"]:
                U = copy.copy(action)
        self._total_output_of_plant_at_MDP_step = [0] * self.number_of_plant_simulation_steps_in_MDP_step
        for i in range(self.number_of_plant_simulation_steps_in_MDP_step):
            for j in range(self.number_of_devices_in_plant):
                device = self.plant[j]
                device.apply_control_signal_in_moment(U[j])
                self._total_output_of_plant_at_MDP_step[i] += device.get_output()
        self._timestamp += self.MDP_discretization_interval
        self._plant_state = self.plant.get_state_vector()
        self._unclaimed_power_current_value = self.unclaimed_power_sensor.get_current_value(self._timestamp, self._plan_generation_mode)
        self._unclaimed_power_forecast = self.unclaimed_power_forecaster.get_forecast(self._timestamp, self._plan_generation_mode)
        self.number_of_MDP_steps_left -= 1
        terminated = (self.number_of_MDP_steps_left == 0)
        reward = self.calculate_reward() #TODO
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info


    def calculate_reward(self):
        # TODO
        pass


    def render(self):
        pass


    def _render_frame(self):
        # use pygame
        pass


    def close(self):
        # quit pygme ets
        pass
