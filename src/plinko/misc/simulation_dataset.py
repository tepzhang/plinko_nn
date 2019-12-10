from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, envs, states):
        """
        envs = [B, env_dim]
        states = [B, max_t, state_dim]
        """
        self.envs = envs
        self.states = states

        self.inputs = states[:, :-1]
        self.targets = states[:, 1:]

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        return {'envs': self.envs[idx], 'states': self.inputs[idx], 'targets': self.targets[idx]}
