from gym.envs.registration import register
from pacman_gym.envs.pacmanInterface import sample_layout

register(
    id='Pacman-v0',
    entry_point='pacman_gym.envs:PacmanEnv',
)


register(
    id='Warehouse-v0',
    entry_point='pacman_gym.envs:WarehouseEnv',
)

register(
    id='Blockworld-v0',
    entry_point='pacman_gym.envs:BlockWorld',
)