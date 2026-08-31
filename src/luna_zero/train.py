"""Vòng lặp huấn luyện Luna Zero.

Ngắt bất cứ lúc nào bằng Ctrl+C: vòng lặp kết thúc bước đang chạy, lưu checkpoint rồi
thoát. Chạy lại đúng lệnh cũ là nó tiếp đúng chỗ — cả optimizer, scheduler lẫn vị trí
trong dữ liệu.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from luna_zero.checkpoint import CheckpointManager, NgatMemMai, TrainState
from luna_zero.config import MODEL, TRAIN, ModelConfig, TrainConfig


# --- lập kế hoạch (không cần torch) -----------------------------------------
@dataclass(frozen=True)
class TrainingPlan:
    tokens_per_step: int
    total_steps: int
    estimated_hours: float

    @property
    def estimated_days(self) -> float:
        return self.estimated_hours / 24


def make_plan(model: ModelConfig = MODEL, train: TrainConfig = TRAIN) -> TrainingPlan:
    tokens_per_step = train.micro_batch_size * train.grad_accum_steps * model.block_size
    total_steps = train.target_tokens // tokens_per_step
    hours = train.target_tokens / train.tokens_per_second_3060 / 3600
    return TrainingPlan(
        tokens_per_step=tokens_per_step, total_steps=total_steps, estimated_hours=hours
    )


def con_lai(plan: TrainingPlan, state: TrainState | None) -> tuple[int, float]:
    """Còn bao nhiêu bước và bao nhiêu giờ nữa, tính từ checkpoint đang có."""
    da_lam = state.step if state else 0
    steps = max(0, plan.total_steps - da_lam)
    return steps, plan.estimated_hours * steps / plan.total_steps


def lr_tai_buoc(step: int, plan: TrainingPlan, train: TrainConfig = TRAIN) -> float:
    """Warmup tuyến tính rồi giảm theo cosine xuống min_lr.

    Warmup là bắt buộc chứ không phải trang trí: những bước đầu gradient rất lớn và
    trạng thái Adam chưa có ước lượng đáng tin, đặt thẳng lr đích thường làm loss nổ.
    """
    if step < train.warmup_steps:
        return train.learning_rate * (step + 1) / train.warmup_steps
    if step >= plan.total_steps:
        return train.min_lr
    tien_do = (step - train.warmup_steps) / max(1, plan.total_steps - train.warmup_steps)
    he_so = 0.5 * (1.0 + math.cos(math.pi * tien_do))
    return train.min_lr + he_so * (train.learning_rate - train.min_lr)


# --- vòng lặp thật ----------------------------------------------------------
def train_loop(
    data_dir: Path,
    tokenizer_path: Path,
    checkpoint_dir: Path,
    model_cfg: ModelConfig = MODEL,
    train_cfg: TrainConfig = TRAIN,
    device: str | None = None,
    max_steps: int | None = None,
    eval_moi: int = 500,
    eval_batches: int = 20,
    log_moi: int = 10,
) -> TrainState:
    """Train từ đầu hoặc chạy tiếp từ checkpoint mới nhất.

    `max_steps` để smoke test chạy vài chục bước rồi dừng.
    """
    import torch

    from luna_zero.config import chon_thiet_bi
    from luna_zero.loader import loader_cho_config
    from luna_zero.model import LunaZeroGPT, format_params
    from luna_zero.pack import kiem_tokenizer_khop

    # Chặn ngay ở khởi động, trước khi đốt bốn ngày: token id trong .bin phải do đúng
    # tokenizer này sinh ra. Sai thì loss vẫn giảm đẹp và chỉ lộ khi sinh văn bản.
    kiem_tokenizer_khop(data_dir, tokenizer_path)

    device = chon_thiet_bi(device)
    plan = make_plan(model_cfg, train_cfg)
    tong_buoc = min(plan.total_steps, max_steps) if max_steps else plan.total_steps

    model = LunaZeroGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.nhom_tham_so_optimizer(train_cfg.weight_decay),
        lr=train_cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    # Suy ra loại thiết bị thay vì viết cứng: bản build torch chỉ có CPU sẽ nổ khi
    # thấy chuỗi "cuda", kể cả lúc autocast đang tắt.
    loai_tb = "cuda" if device.startswith("cuda") else "cpu"
    dung_amp = loai_tb == "cuda"
    scaler = torch.amp.GradScaler(device=loai_tb, enabled=dung_amp)

    manager = CheckpointManager(checkpoint_dir, max_keep=train_cfg.max_checkpoints_keep)
    da_co = manager.load_latest()
    if da_co is None:
        state = TrainState(step=0, tokens_seen=0, data_position=0)
        print(f"Model    : {format_params(model.so_tham_so())} tham số, thiết bị {device}")
        print("Trạng thái: train từ đầu")
    else:
        state, payload = da_co
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if dung_amp and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        print(f"Trạng thái: chạy tiếp từ bước {state.step:,}")

    # data_position là SỐ CỬA SỔ đã tiêu thụ trong hoán vị, không phải số token.
    # Truyền nó vào loader chính là thứ làm cho việc chạy tiếp chính xác.
    train_loader = loader_cho_config(
        data_dir, "train", model_cfg, device, vi_tri=state.data_position
    )
    val_loader = loader_cho_config(data_dir, "val", model_cfg, device, seed=0)
    cua_so_moi_buoc = train_cfg.micro_batch_size * train_cfg.grad_accum_steps

    @torch.no_grad()
    def do_val() -> float:
        model.eval()
        tong = 0.0
        for _ in range(eval_batches):
            x, y = val_loader.lay_batch(train_cfg.micro_batch_size)
            with torch.autocast(device_type=loai_tb, dtype=torch.bfloat16, enabled=dung_amp):
                tong += model(x, y).loss.item()
        model.train()
        return tong / eval_batches

    model.train()
    t0 = time.perf_counter()
    with NgatMemMai() as ngat:
        for step in range(state.step, tong_buoc):
            lr = lr_tai_buoc(step, plan, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_gop = 0.0
            for _ in range(train_cfg.grad_accum_steps):
                x, y = train_loader.lay_batch(train_cfg.micro_batch_size)
                with torch.autocast(device_type=loai_tb, dtype=torch.bfloat16, enabled=dung_amp):
                    loss = model(x, y).loss / train_cfg.grad_accum_steps
                scaler.scale(loss).backward()
                loss_gop += loss.item()

            # unscale trước khi cắt gradient, nếu không thì ngưỡng cắt bị nhân với
            # hệ số scale và phép cắt thành vô nghĩa.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            state = TrainState(
                step=step + 1,
                tokens_seen=state.tokens_seen + plan.tokens_per_step,
                data_position=state.data_position + cua_so_moi_buoc,
                best_val_loss=state.best_val_loss,
            )

            if (step + 1) % log_moi == 0:
                giay = time.perf_counter() - t0
                tps = state.tokens_seen and plan.tokens_per_step * log_moi / giay
                print(
                    f"bước {step + 1:>7,}/{tong_buoc:,} | loss {loss_gop:.4f} "
                    f"| lr {lr:.2e} | {tps / 1e3:.1f}k token/s",
                    flush=True,
                )
                t0 = time.perf_counter()

            den_luc_luu = (step + 1) % train_cfg.save_every_steps == 0
            if den_luc_luu or ngat.duoc_yeu_cau_dung or (step + 1) == tong_buoc:
                val_loss = do_val()
                tot_hon = val_loss < state.best_val_loss
                state = TrainState(
                    step=state.step,
                    tokens_seen=state.tokens_seen,
                    data_position=state.data_position,
                    best_val_loss=min(val_loss, state.best_val_loss),
                )
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if dung_amp else None,
                    "model_cfg": model_cfg.__dict__,
                }
                manager.save(state, payload, is_best=tot_hon)
                print(f"  val loss {val_loss:.4f}{' (tốt nhất)' if tot_hon else ''} — đã lưu")
                if ngat.duoc_yeu_cau_dung:
                    print("Đã dừng theo yêu cầu. Chạy lại lệnh cũ để tiếp tục.")
                    break
    return state


def main() -> None:
    plan = make_plan()
    print(f"Token mỗi bước : {plan.tokens_per_step:,}")
    print(f"Tổng số bước   : {plan.total_steps:,}")
    print(f"Thời gian ước  : {plan.estimated_days:.1f} ngày")
    print(
        f"Lưu checkpoint : mỗi {TRAIN.save_every_steps} bước, giữ {TRAIN.max_checkpoints_keep} bản"
    )
    print("Chạy thật bằng: python scripts/train_model.py")


if __name__ == "__main__":
    main()
