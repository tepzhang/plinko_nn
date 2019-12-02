from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, envs, states):
        """
        envs = [B, env_dim]
        states = [B, max_t, state_dim]
        """
        self.envs = envs
        self.states = states

        # # print('state shape is ', states.shape)
        # if states.shape[2] == 3:
        #     # print('add col in states')
        #     self.inputs = states[:, :-1, :3]
        # else:
        #     self.inputs = states[:, :-1, :2]
        # self.targets = states[:, 1:, :2]
        self.inputs = states[:, :-1]
        self.targets = states[:, 1:]

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        return {'envs': self.envs[idx], 'states': self.inputs[idx], 'targets': self.targets[idx]}
