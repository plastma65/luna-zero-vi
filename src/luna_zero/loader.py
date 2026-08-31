"""Lấy batch từ train.bin / val.bin bằng memmap.

Duyệt TUẦN TỰ trên một hoán vị cố định của các cửa sổ, không lấy mẫu ngẫu nhiên có
hoàn lại. Hai lý do, cả hai đều quan trọng:

1. **Độ phủ.** Lấy ngẫu nhiên có hoàn lại 2,2 tỷ token từ corpus 2,26 tỷ chỉ chạm tới
   khoảng 62% corpus — gần 38% dữ liệu không bao giờ được nhìn, trong khi phần khác bị
   lặp. Duyệt hoán vị thì mỗi cửa sổ được thấy đúng một lần trước khi sang vòng mới.
2. **Chạy tiếp chính xác.** Vị trí trong hoán vị là một số nguyên, lưu vào checkpoint
   được. Nhờ vậy `data_position` là thứ thật sự điều khiển bộ nạp, chứ không phải con
   số ghi cho có.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from luna_zero import config
from luna_zero.config import MODEL, ModelConfig


class BatchLoader:
    """Sinh cặp (x, y) cho một split.

    `y` là `x` DỊCH SANG PHẢI ĐÚNG MỘT VỊ TRÍ. Đây là chỗ sai kinh điển: lệch sai một
    bước thì model học dự đoán token hiện tại thay vì token kế tiếp, loss lao xuống gần 0
    và trông như thành công vang dội. Có test khoá quan hệ dịch này.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        block_size: int = MODEL.block_size,
        device: str | None = None,
        seed: int = 1337,
        vi_tri: int = 0,
    ) -> None:
        path = Path(data_dir) / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}. Chạy scripts/dong_goi.py trước.")
        self.data = np.memmap(path, dtype=config.DATA.token_dtype, mode="r")
        self.block_size = block_size
        self.device = device or config.chon_thiet_bi()
        self.seed = seed

        # Cửa sổ không chồng lấn, mỗi cửa sổ cần block_size+1 token để cắt cả x lẫn y.
        self.n_cua_so = (self.data.size - 1) // block_size
        if self.n_cua_so < 1:
            raise ValueError(
                f"{path.name} chỉ có {self.data.size:,} token, "
                f"cần hơn {block_size + 1:,} để cắt một cửa sổ"
            )
        self.vi_tri = int(vi_tri)
        self._nap_hoan_vi()

    def _nap_hoan_vi(self) -> None:
        """Hoán vị của vòng hiện tại. Seed = seed gốc + số vòng, nên mỗi vòng có thứ tự
        khác nhau nhưng tái lập được từ checkpoint."""
        self.vong = self.vi_tri // self.n_cua_so
        rng = np.random.default_rng(self.seed + self.vong)
        self.hoan_vi = rng.permutation(self.n_cua_so)

    def __len__(self) -> int:
        return self.n_cua_so

    def lay_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        chi_so = []
        for _ in range(batch_size):
            if self.vi_tri // self.n_cua_so != self.vong:
                self._nap_hoan_vi()  # hết một vòng, trộn lại
            chi_so.append(int(self.hoan_vi[self.vi_tri % self.n_cua_so]))
            self.vi_tri += 1

        starts = [i * self.block_size for i in chi_so]
        # astype(int64): token lưu uint16, nhưng nn.Embedding đòi chỉ số int64.
        x = np.stack([self.data[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.block_size] for s in starts]).astype(np.int64)
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        if self.device.startswith("cuda"):
            # pin_memory + non_blocking: chép sang GPU chồng lấn với tính toán.
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt, yt = xt.to(self.device), yt.to(self.device)
        return xt, yt

    def dat_vi_tri(self, vi_tri: int) -> None:
        """Nhảy tới một vị trí trong hoán vị — dùng khi chạy tiếp từ checkpoint."""
        self.vi_tri = int(vi_tri)
        self._nap_hoan_vi()


def loader_cho_config(
    data_dir: Path,
    split: str,
    cfg: ModelConfig = MODEL,
    device: str | None = None,
    seed: int = 1337,
    vi_tri: int = 0,
) -> BatchLoader:
    return BatchLoader(
        data_dir, split, block_size=cfg.block_size, device=device, seed=seed, vi_tri=vi_tri
    )
