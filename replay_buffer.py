from collections import deque


class StateBuffer:

    def __init__(self, max_len, initial_frames, state_len):
        self.state_len = state_len
        self.buf = deque(initial_frames, maxlen=max_len + self.state_len - 1)

    def __len__(self):
        if isinstance(self.buf[-1], list):
            return len(self.buf)
        return max(0, len(self.buf) - self.state_len + 1)

    def __getitem__(self, i):
        item = []
        for j in range(i, i + self.state_len):
            frame = self.buf[j]
            if isinstance(frame, list):
                item.extend(frame[:i + self.state_len - j])
                break
            else:
                item.append(frame)
        return item

    def new_episode(self, initial_frames):
        last_state = list(reversed([self.buf.pop() for _ in range(self.state_len)]))
        self.buf.append(last_state)
        self.buf.extend(initial_frames)

    def append(self, frame):
        self.buf.append(frame)


class ReplayBuffer:

    def __init__(self, max_len, initial_frames, sate_len):
        self.buf = deque(maxlen=max_len)
        self.state_transition_buf = StateBuffer(max_len + 1, initial_frames, sate_len + 1)

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        action, reward = self.buf[i]
        state_transition = self.state_transition_buf[i]
        return action, reward, state_transition

    def append(self, transition):
        action, reward, frame = transition
        self.buf.append((action, reward))
        self.state_transition_buf.append(frame)

    def new_episode(self, initial_frames):
        self.state_transition_buf.new_episode(initial_frames)
