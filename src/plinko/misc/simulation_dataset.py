from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, envs, states,outv = False):
        """
        envs = [B, env_dim]
        states = [B, max_t, state_dim]
        """
        self.envs = envs
        self.states = states

        if outv:
            self.inputs = states[:, :-1, :4]
            self.targets = states[:, 1:, :4]
        else:
            self.inputs = states[:, :-1, :2]
            self.targets = states[:, 1:, :2]

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, idx):
        return {'envs': self.envs[idx], 'states': self.inputs[idx], 'targets': self.targets[idx]}
