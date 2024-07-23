from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
)

register(
    id="gym_examples/distributed_offtake_of_the_unclaimed_power-v0",
    entry_point="gym_examples.envs:Environment",
)
