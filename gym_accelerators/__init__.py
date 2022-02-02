from gym.envs.registration import register

register(
    id='Surrogate_Accelerator-v1',
    entry_point='gym_accelerators.envs:Surrogate_Accelerator_v1',
)
register(
    id='ExaBooster-v1',
    entry_point='gym_accelerators.envs:ExaBooster_v1',
)

register(
    id='ExaBooster-v2',
    entry_point='gym_accelerators.envs:ExaBooster_v2',
)