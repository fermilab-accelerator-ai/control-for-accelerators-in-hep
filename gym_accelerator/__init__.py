from gym.envs.registration import register

register(
    id='Data_Accelerator-v0',
    entry_point='gym_accelerator.envs:Data_Accelerator',
)

register(
    id='Emulator_Accelerator-v0',
    entry_point='gym_accelerator.envs:Emulator_Accelerator',
)

register(
    id='Surrogate_Accelerator-v0',
    entry_point='gym_accelerator.envs:Surrogate_Accelerator',
)

register(
    id='Surrogate_Accelerator-v2',
    entry_point='gym_accelerator.envs:Surrogate_Accelerator_v2',
)

register(
    id='Surrogate_Accelerator-v3',
    entry_point='gym_accelerator.envs:Surrogate_Accelerator_v3',
)