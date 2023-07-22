def clamp_reward(reward: float, min_reward: float = -1.0, max_reward: float = 1.0) -> float:
    return max(min_reward, min(max_reward, reward))
