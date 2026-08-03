"""10 帧观测历史缓冲区。

IsaacLab 训练时 Encoder 输入是 oldest-to-newest 的 `[10, 28]` 历史窗口。
第一次 reset 后还没有真实历史，因此用首帧观测复制填满 10 帧，和训练端
CircularBuffer 的首次写入语义保持一致。
"""

from __future__ import annotations

import numpy as np


class ObservationHistory:
    """固定长度、oldest-to-newest 排列的观测历史。"""

    def __init__(self, length: int, observation_dim: int) -> None:
        if length <= 0 or observation_dim <= 0:
            raise ValueError("History dimensions must be positive.")
        self._data = np.zeros((length, observation_dim), dtype=np.float32)
        self._initialized = False

    @property
    def data(self) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("Observation history has not been initialized.")
        return self._data

    def reset(self) -> None:
        self._data.fill(0.0)
        self._initialized = False

    def append(self, observation: np.ndarray) -> None:
        """追加一帧 28 维观测；首次追加会复制填满整个窗口。"""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self._data.shape[1],):
            raise ValueError(f"Expected observation shape {(self._data.shape[1],)}, got {observation.shape}.")
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Observation contains NaN or Inf.")
        if not self._initialized:
            self._data[:] = observation     # 第一次写入：全部填充同一帧
            self._initialized = True
            return
        # 维持 oldest-to-newest：丢弃最老帧，新观测写入最后一帧。
        self._data[:-1] = self._data[1:].copy()
        self._data[-1] = observation

    def batched(self) -> np.ndarray:
        """返回 ONNX 需要的 batch 形状 `[1, history_len, obs_dim]`。"""
        return np.ascontiguousarray(self.data[None, ...], dtype=np.float32)

'''
数据存储方式（滑动窗口）：
    时间 →
    初始化:  [obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0]
    第1步:   [obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs1]
    第2步:   [obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs0, obs1, obs2]
    ...
    第10步:  [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10]
'''
