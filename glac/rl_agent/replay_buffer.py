# replay_buffer.py
import numpy as np
import jax.tree_util as jtu
from collections import deque
import random
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from .data import Rollout
from glac.utils.utils import merge01
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        # self.capacity = capacity
        # self.ptr = 0
        # self.size = 0
        #self.buffer = deque(maxlen=capacity)
        self._size = capacity
        self._buffer = None # 仍然是 Pytree
        # self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        # self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        # self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        # self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, rollout: Rollout):
        #transition = (state, action, reward, next_state, done)
        # transition = (action, reward, done)
        # self.buffer.append(transition)

        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)
    def sample(self, batch_size: int) -> Rollout:
        #idx = np.random.randint(0, self._buffer.length, batch_size)
        #rollout = self.get_data(idx)
        #rollout_batch = jtu.tree_map(lambda x: merge01(x), rollout)
        rollout = jtu.tree_map(lambda x: merge01(x), self._buffer)
        idx = np.random.randint(0, self.length, batch_size)
        rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
        
        return rollout_batch

    def get_data(self, idx: np.ndarray) -> Rollout:
        return jtu.tree_map(lambda x: x[idx], self._buffer)
    
    def __len__(self):
        return len(self.buffer)
    @property
    def length(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.n_data


class PyTreeReplayBuffer:
    def __init__(self, capacity: int, dummy_input):
    
        self.capacity = int(capacity)
        self.edge_capacity = int(2e5) 
        self.ptr = 0
        self.edge_pointer = 0
        self.size = 0
        self.edge_size = 0
       
        flat_input, self.tree_def = jtu.tree_flatten(dummy_input)
        

        self.buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
        self.edge_buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
    def add_batch(self, batch_data):
        
        flat_batch_data = jtu.tree_leaves(batch_data)
        num_to_add = flat_batch_data[0].shape[0]
        
        
        if self.ptr + num_to_add <= self.capacity:
           
            idxs = np.arange(self.ptr, self.ptr + num_to_add)
            
            
            for i, leaf_batch in enumerate(flat_batch_data):
                self.buffers[i][idxs] = leaf_batch
        else:
            
            num_part1 = self.capacity - self.ptr
            idxs_part1 = np.arange(self.ptr, self.capacity)
            
            
            num_part2 = num_to_add - num_part1
            idxs_part2 = np.arange(0, num_part2)

            flat_batch_data = jtu.tree_leaves(batch_data)
            for i, leaf_batch in enumerate(flat_batch_data):
               
                self.buffers[i][idxs_part1] = leaf_batch[:num_part1]
               
                self.buffers[i][idxs_part2] = leaf_batch[num_part1:]
        
       
        self.ptr = (self.ptr + num_to_add) % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)
    
    def add_edge(self, edge_N, episode_transitions):
        
        if edge_N != -1:
            episode_edge_transitions = jtu.tree_map(
                lambda x: x[0:edge_N],
                episode_transitions
            )
            flat_batch_data = jtu.tree_leaves(episode_edge_transitions)
            num_to_add = flat_batch_data[0].shape[0]
        
           
            if self.edge_pointer + num_to_add <= self.edge_capacity:
               
                idxs = np.arange(self.edge_pointer, self.edge_pointer + num_to_add)
                for i, leaf_batch in enumerate(flat_batch_data):
                    self.edge_buffers[i][idxs] = leaf_batch
            else:
                
                num_part1 = self.edge_capacity - self.edge_pointer
                idxs_part1 = np.arange(self.edge_pointer, self.edge_capacity)
                
               
                num_part2 = num_to_add - num_part1
                idxs_part2 = np.arange(0, num_part2)

                for i, leaf_batch in enumerate(flat_batch_data):
                   
                    self.edge_buffers[i][idxs_part1] = leaf_batch[:num_part1]
                   
                    self.edge_buffers[i][idxs_part2] = leaf_batch[num_part1:]
            
           
            self.edge_pointer = (self.edge_pointer + num_to_add) % self.edge_capacity
            self.edge_size = min(self.edge_size + num_to_add, self.edge_capacity)
            
    def add(self, data):
        
        flat_data = jtu.tree_leaves(data)
        
       
        for i, leaf in enumerate(flat_data):
            self.buffers[i][self.ptr] = leaf
            
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        
        sampled_leaves = [buf[idxs] for buf in self.buffers]
        
       
        main_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        edge_batch = None
        if self.edge_size > batch_size:
            edge_idx = np.random.randint(0, self.edge_size, size=batch_size)
            sampled_leaves = [buf[edge_idx] for buf in self.edge_buffers]
            edge_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        return main_batch, edge_batch
        
    def __len__(self):
        return self.size