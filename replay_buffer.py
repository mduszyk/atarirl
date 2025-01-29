from collections import deque


class FramesBuffer:

    def __init__(self, max_len, initial_frames, item_len):
        self.item_len = item_len
        self.buf = deque(initial_frames, maxlen=max_len + self.item_len - 1)

    def __len__(self):
        if isinstance(self.buf[-1], list):
            return len(self.buf)
        return max(0, len(self.buf) - self.item_len + 1)

    def __getitem__(self, i):
        item = []
        for j in range(i, i + self.item_len):
            frame = self.buf[j]
            if isinstance(frame, list):
                item.extend(frame[:i + self.item_len - j])
                break
            else:
                item.append(frame)
        return item

    def new_episode(self, initial_frames):
        last_item = list(reversed([self.buf.pop() for _ in range(self.item_len)]))
        self.buf.append(last_item)
        self.buf.extend(initial_frames)

    def append(self, frame):
        self.buf.append(frame)


class ReplayBuffer:

    def __init__(self, max_len, initial_frames, state_len):
        assert len(initial_frames) == state_len
        self.action_reward_buf = deque(maxlen=max_len)
        self.state_transition_buf = FramesBuffer(max_len, initial_frames, state_len + 1)
        self.state_len = state_len

    def __len__(self):
        assert len(self.action_reward_buf) == len(self.state_transition_buf)
        return len(self.action_reward_buf)

    def __getitem__(self, i):
        action, reward = self.action_reward_buf[i]
        state_transition = self.state_transition_buf[i]
        return action, reward, state_transition

    def append(self, action, reward, frame):
        self.action_reward_buf.append((action, reward))
        self.state_transition_buf.append(frame)

    def new_episode(self, initial_frames):
        assert len(initial_frames) == self.state_len
        self.state_transition_buf.new_episode(initial_frames)
