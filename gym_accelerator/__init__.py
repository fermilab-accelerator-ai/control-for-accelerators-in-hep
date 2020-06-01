from gym.envs.registration import register

register(
    id='LSTM_Accelerator-v0',
    entry_point='gym_accelerator.envs:LSTM_Accelerator',
)
register(
    id='Data_Accelerator-v0',
    entry_point='gym_accelerator.envs:Data_Accelerator',
)
