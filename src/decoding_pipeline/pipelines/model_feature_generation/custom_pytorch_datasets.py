from torch.utils.data import Dataset

class CorticomExperimentDataset(Dataset):
    def __init__(self, trial_filenames_dict, trial_labels_list, trial_loading_dict):
        self.trial_filenames_dict = trial_filenames_dict
        self.trial_labels_list = trial_labels_list
        self.trial_loading_dict = trial_loading_dict

    def __len__(self):
        return len(self.trial_filenames_dict)

    def __getitem__(self, idx):
        filename = self.trial_filenames_dict[idx]
        
        return self.trial_loading_dict[filename](), self.trial_labels_list[idx]

