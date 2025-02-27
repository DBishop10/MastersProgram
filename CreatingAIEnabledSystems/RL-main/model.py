import numpy as np
import pandas as pd
import pickle

class EmailCampaign:
    def __init__(self, training_data=None, actions=None, load_from=None):
        if load_from:
            self.load_model(load_from)
        else:
            self.training_data = training_data
            self.actions = actions.unique()  # Assuming actions are passed directly
            self.q_table = np.zeros((len(self.training_data['State'].unique()), len(self.actions)))
            # Creating mappings from state and action to index
            self.state_to_index = {state: idx for idx, state in enumerate(self.training_data['State'].unique())}
            self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
    
    def update_q_table(self, state, action, reward, alpha, gamma):
        state_idx = self.state_to_index[state]
        action_idx = self.action_to_index[action]
        current_q = self.q_table[state_idx, action_idx]
        next_max_q = np.max(self.q_table[state_idx])  # Assuming no next state is known
        self.q_table[state_idx, action_idx] = current_q + alpha * (reward + gamma * next_max_q - current_q)
        
    def convert_to_state(self, gender, typed, age, tenure):
        age_bins = [18, 25, 35, 50, 65]
        age_labels = ['18-24', '25-34', '35-49', '50-64']
        tenure_bins = [0, 5, 10, 15, 20, 40] 
        tenure_labels = ['0-4', '5-9', '10-14', '15-20', '21+']
        age_group_series = pd.cut([age], bins=age_bins, labels=age_labels, right=False)
        tenure_group_series = pd.cut([tenure], bins=tenure_bins, labels=tenure_labels, right=False)
        age_group = age_group_series[0]
        tenure_group = tenure_group_series[0]
        state = f"{gender}-{typed}-{age_group}-{tenure_group}"
        return state
        
    def save_model(self, filename_prefix):
        # Save the Q-table
        np.save(f'{filename_prefix}_qtable.npy', self.q_table)
        # Save the mappings
        with open(f'{filename_prefix}_state_to_index.pickle', 'wb') as handle:
            pickle.dump(self.state_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{filename_prefix}_action_to_index.pickle', 'wb') as handle:
            pickle.dump(self.action_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename_prefix):
        # Load the Q-table
        self.q_table = np.load(f'{filename_prefix}_qtable.npy')
        # Load the mappings
        with open(f'{filename_prefix}_state_to_index.pickle', 'rb') as handle:
            self.state_to_index = pickle.load(handle)
        with open(f'{filename_prefix}_action_to_index.pickle', 'rb') as handle:
            self.action_to_index = pickle.load(handle)
        self.actions = np.array(list(self.action_to_index.keys()))  # Reconstruct actions array
        
    def get_best_action_for_state(self, state_index):
        # Return the index of the action with the highest Q-value for the given state index
        state_idx = self.state_to_index[state_index]
        best_action_index = np.argmax(self.q_table[state_idx])
        return self.actions[best_action_index]