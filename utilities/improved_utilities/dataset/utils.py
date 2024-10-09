import sys
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
import os
from improved_utilities.img import resize
import logging
import tqdm

assert sys.maxsize > (
    2**32
), "you need to be on 64 bit system to store > 2GB experience for your q-transformer agent"


def exists(v):
    return v is not None


def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t


class ReplayMemoryDataset(Dataset):
    def __init__(self, iteration: int, path: str, num_timesteps):
        super().__init__()
        assert num_timesteps >= 1, "num_timesteps must be at least 1"
        self.is_single_timestep = num_timesteps == 1
        self.num_timesteps = num_timesteps

        self.states = open_memmap(
            os.path.join(path, f"states_{iteration}.memmap.npy"),
            dtype="float32",
            mode="r",
        )
        self.actions = open_memmap(
            os.path.join(path, f"actions_{iteration}.memmap.npy"),
            dtype="float32",
            mode="r",
        )
        self.rewards = open_memmap(
            os.path.join(path, f"rewards_{iteration}.memmap.npy"),
            dtype="float32",
            mode="r",
        )
        self.dones = open_memmap(
            os.path.join(path, f"dones_{iteration}.memmap.npy"), dtype="bool", mode="r"
        )

        self.episode_length = (self.dones.cumsum(axis=-1) == 0).sum(axis=-1) + 1
        self.num_episodes, self.max_episode_len = self.dones.shape

        trainable_episode_indices = self.episode_length >= num_timesteps

        assert self.dones.size > 0, "no episodes found"

        self.num_episodes, self.max_episode_len = self.dones.shape

        timestep_arange = torch.arange(self.max_episode_len)

        timestep_indices = torch.stack(
            torch.meshgrid(torch.arange(self.num_episodes), timestep_arange), dim=-1
        )
        trainable_mask = timestep_arange < (
            (torch.from_numpy(self.episode_length) - num_timesteps).unsqueeze(1)
        )
        self.indices = timestep_indices[trainable_mask]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        episode_index, timestep_index = self.indices[idx]
        timestep_slice = slice(timestep_index, (timestep_index + self.num_timesteps))
        timestep_slice_next = slice(
            timestep_index + 1, (timestep_index + self.num_timesteps) + 1
        )
        state = self.states[episode_index, timestep_slice].copy()
        action = self.actions[episode_index, timestep_slice].copy()
        reward = self.rewards[episode_index, timestep_slice].copy()
        done = self.dones[episode_index, timestep_slice].copy()
        next_state = self.states[episode_index, timestep_slice_next].copy()
        next_action = self.actions[episode_index, timestep_slice_next].copy()
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "next_state": next_state,
            "next_action": next_action,
        }


class SampleData:
    def __init__(
        self,
        iteration: int,
        path: str,  # where you want to save
        num_episodes: int,
        max_num_steps_per_episode: int,
        state_shape: tuple,
        action_shape: tuple,
    ):
        super().__init__()
        if not os.path.exists(path):
            os.makedirs(path)

        prec_shape = (num_episodes, max_num_steps_per_episode)
        self.num_episodes=num_episodes
        self.max_num_steps_per_episode=max_num_steps_per_episode
        self.states = open_memmap(
            str(os.path.join(path, f"states_{iteration}.memmap.npy")),
            dtype="float32",
            mode="w+",
            shape=(*prec_shape, *state_shape),
        )
        self.actions = open_memmap(
            os.path.join(path, f"actions_{iteration}.memmap.npy"),
            dtype="float32",
            mode="w+",
            shape=(*prec_shape, *action_shape),
        )
        self.rewards = open_memmap(
            os.path.join(path, f"rewards_{iteration}.memmap.npy"),
            dtype="float32",
            mode="w+",
            shape=prec_shape,
        )
        self.dones = open_memmap(
            os.path.join(path, f"dones_{iteration}.memmap.npy"),
            dtype="bool",
            mode="w+",
            shape=prec_shape,
        )



    def start_sample_game(self, env,transform=None):
        from rich.progress import track
        for episode in range(self.num_episodes):
            done = False
            obs = resize(env.reset(),64)
            logging.info(f"Episode {episode} started.")
            for step in track(range(self.max_num_steps_per_episode)):
                last_step = step == (self.max_num_steps_per_episode - 1)
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                next_state = resize(next_state,64)
                done = done | last_step
                if transform is not None:
                    obs=transform(obs)
                self.states[episode, step] = obs
                self.actions[episode, step] = action
                self.rewards[episode, step] = reward
                self.dones[episode, step] = done
                
                
                # if done, move onto next episode
                if done:
                    break
                
                # set next state
                obs = next_state
            
            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()
        
        del self.states
        del self.actions
        del self.rewards
        del self.dones
        


        logging.info("Completed all episodes.")


    # # @torch.no_grad()
    # def start_smple(self, env):
    #     for episode in range(self.num_episodes):
    #         print(f"episode {episode}")
    #         curr_state, log = env.reset()
    #         curr_state = self.transform(curr_state)
    #         for step in track(range(self.max_num_steps_per_episode)):
    #             last_step = step == (self.max_num_steps_per_episode - 1)

    #             action = self.env.action_space.sample()
    #             next_state, reward, termiuted, tuned, log = self.env.step(action)
    #             next_state = self.transform(next_state)
    #             done = termiuted | tuned | last_step
    #             # store memories using memmap, for later reflection and learning
    #             self.states[episode, step] = curr_state
    #             self.actions[episode, step] = action
    #             self.rewards[episode, step] = reward
    #             self.dones[episode, step] = done
    #             # if done, move onto next episode
    #             if done:
    #                 break
    #             # set next state
    #             curr_state = next_state

    #         self.states.flush()
    #         self.actions.flush()
    #         self.rewards.flush()
    #         self.dones.flush()

    #     del self.states
    #     del self.actions
    #     del self.rewards
    #     del self.dones
    #     self.memories_dataset_folder.resolve()
    #     print(f"completed")

    def transformer_frome_diengine(self, path):
        collected_episodes = torch.load(path)
        for episode_idx, episode in enumerate(collected_episodes):
            for step_idx, step in enumerate(episode):
                self.states[episode_idx, step_idx] = step["obs"]
                self.actions[episode_idx, step_idx] = step["action"]
                self.rewards[episode_idx, step_idx] = step["reward"]
                self.dones[episode_idx, step_idx] = step["done"]
            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()
        del self.states
        del self.actions
        del self.rewards
        del self.dones
        print(f"completed")
