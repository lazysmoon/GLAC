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
        self._buffer = None # still a Pytree
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

        # if self.buffer_ is None:
        #     self.buffer_ = jax2np(next_state)
        # else:
        #     self.buffer_ = tree_merge([self.buffer_, jax2np(next_state)])
        # if self.buffer_.length > self._size:
        #     self.buffer_ = jtu.tree_map(lambda x: x[-self._size:], self.buffer_)
        # # self.states[self.ptr] = state
        # self.actions[self.ptr] = action
        # self.rewards[self.ptr] = reward
        # #self.next_states[self.ptr] = next_state
        # self.dones[self.ptr] = done

        # self.ptr = (self.ptr + 1) % self.capacity
        # self.size = min(self.size + 1, self.capacity)

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
    """
    A generic experience replay buffer that can store and sample data with an
    arbitrary PyTree structure. For example, it can store transitions such as
    (GraphsTuple, action, reward, ...).
    """
    def __init__(self, capacity: int, dummy_input):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum capacity of the buffer (number of transitions).
            dummy_input: A "dummy" or "template" object with exactly the same
                         PyTree structure as a single data sample to be stored,
                         e.g. a transition tuple.
        """
        self.capacity = int(capacity)
        self.edge_capacity = int(2e5)  # configurable
        self.ptr = 0
        self.edge_pointer = 0
        self.size = 0
        self.edge_size = 0
        # --- Core initialization ---
        # 1. Flatten the dummy input to get the list of leaves and the PyTree structure
        flat_input, self.tree_def = jtu.tree_flatten(dummy_input)

        # 2. Create a corresponding NumPy storage array for each leaf.
        #    The first dimension of every storage array is `capacity`.
        self.buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
        self.edge_buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
    def add_batch(self, batch_data):
        """
        Add a batch of data to the buffer.

        Args:
            batch_data: A batched PyTree whose leaves all have a leading batch
                        dimension, e.g. (100, ...).
        """
        # 1. Determine how many samples to add this time
        flat_batch_data = jtu.tree_leaves(batch_data)
        num_to_add = flat_batch_data[0].shape[0]

        # 2. Compute the index range to write to
        #    This logic handles the wrap-around case when the buffer is full
        if self.ptr + num_to_add <= self.capacity:
            # a. Enough space: write directly
            idxs = np.arange(self.ptr, self.ptr + num_to_add)


            for i, leaf_batch in enumerate(flat_batch_data):
                self.buffers[i][idxs] = leaf_batch
        else:
            # b. Not enough space: write in two parts (ring buffer)
            #    Part 1: fill the tail
            num_part1 = self.capacity - self.ptr
            idxs_part1 = np.arange(self.ptr, self.capacity)

            #    Part 2: write from the beginning
            num_part2 = num_to_add - num_part1
            idxs_part2 = np.arange(0, num_part2)

            flat_batch_data = jtu.tree_leaves(batch_data)
            for i, leaf_batch in enumerate(flat_batch_data):
                # Write part 1
                self.buffers[i][idxs_part1] = leaf_batch[:num_part1]
                # Write part 2
                self.buffers[i][idxs_part2] = leaf_batch[num_part1:]

        # 3. Update the pointer and size
        self.ptr = (self.ptr + num_to_add) % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)
    
    def add_edge(self, edge_N, episode_transitions):
        """
        Called at the end of an episode to process the whole trajectory and store
        it in the buffer.
        is_edge_fn: a function that takes state s and returns a bool (whether it is
        in the edge region).
        """
        # #a = episode_transitions[0].env_states.edge_mask
        # edge_indices = np.where(episode_transitions[0].env_states.edge_mask == 1)[0]
        
        # if edge_indices.shape[0] > 0:
        #     episode_edge_transitions = jtu.tree_map(
        #         lambda x: x[edge_indices],
        #         episode_transitions
        #     )
        if edge_N != -1:
            episode_edge_transitions = jtu.tree_map(
                lambda x: x[0:edge_N],
                episode_transitions
            )
            flat_batch_data = jtu.tree_leaves(episode_edge_transitions)
            num_to_add = flat_batch_data[0].shape[0]
        
            # 2. Compute the index range to write to
            #    This logic handles the wrap-around case when the buffer is full
            if self.edge_pointer + num_to_add <= self.edge_capacity:
                # a. Enough space: write directly
                idxs = np.arange(self.edge_pointer, self.edge_pointer + num_to_add)
                for i, leaf_batch in enumerate(flat_batch_data):
                    self.edge_buffers[i][idxs] = leaf_batch
            else:
                # b. Not enough space: write in two parts (ring buffer)
                #    Part 1: fill the tail
                num_part1 = self.edge_capacity - self.edge_pointer
                idxs_part1 = np.arange(self.edge_pointer, self.edge_capacity)

                #    Part 2: write from the beginning
                num_part2 = num_to_add - num_part1
                idxs_part2 = np.arange(0, num_part2)

                for i, leaf_batch in enumerate(flat_batch_data):
                    # Write part 1
                    self.edge_buffers[i][idxs_part1] = leaf_batch[:num_part1]
                    # Write part 2
                    self.edge_buffers[i][idxs_part2] = leaf_batch[num_part1:]

            # 3. Update the pointer and size
            self.edge_pointer = (self.edge_pointer + num_to_add) % self.edge_capacity
            self.edge_size = min(self.edge_size + num_to_add, self.edge_capacity)
    def add(self, data):
        """
        Add a single data sample (e.g. one transition) to the buffer.

        Args:
            data: A PyTree object with exactly the same structure as the
                  dummy_input used at initialization.
        """
        # 1. Flatten the input data into a list of leaves
        flat_data = jtu.tree_leaves(data)

        # 2. Iterate over the leaves and store each into its NumPy storage array
        for i, leaf in enumerate(flat_data):
            self.buffers[i][self.ptr] = leaf

        # 3. Update the pointer and current size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of data from the buffer.

        Args:
            batch_size: Batch size.

        Returns:
            A batched PyTree with the same structure as a single sample, but
            with every leaf having an extra leading batch dimension.
        """
        # 1. Generate random indices
        idxs = np.random.randint(0, self.size, size=batch_size)

        # 2. Slice data out of each NumPy storage array by index
        sampled_leaves = [buf[idxs] for buf in self.buffers]

        # 3. Reassemble the sampled leaves into a PyTree using the stored structure
        main_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        edge_batch = None
        if self.edge_size > batch_size:
            edge_idx = np.random.randint(0, self.edge_size, size=batch_size)
            sampled_leaves = [buf[edge_idx] for buf in self.edge_buffers]
            edge_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        return main_batch, edge_batch
        
    def __len__(self):
        return self.size