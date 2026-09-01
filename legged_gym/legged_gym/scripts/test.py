import torch
import os


# 这里换成你的 IK 网络 .pt 路径
IK_PT_PATH = "../envs/go2_arm/arm_ik_net.pt"


# ============================================================
# 这里必须和你训练 IK 时的网络结构完全一致
# 如果你训练脚本里已经有 NeuralIKNet，就直接 import 进来
# 例如：
# from train_neural_ik import NeuralIKNet
# ============================================================

class NeuralIKNet(torch.nn.Module):
    def __init__(self, in_dim=6, out_dim=6, hidden_dims=(256, 256, 256)):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, h))
            layers.append(torch.nn.ELU())
            last_dim = h
        layers.append(torch.nn.Linear(last_dim, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in [
        "model_state_dict",
        "state_dict",
        "ik_state_dict",
        "net_state_dict",
    ]:
        if key in ckpt:
            return ckpt[key]

    # 如果整个 ckpt 本身就是 state_dict
    maybe_state_dict = {
        k: v for k, v in ckpt.items()
        if isinstance(v, torch.Tensor)
    }
    if len(maybe_state_dict) > 0:
        return maybe_state_dict

    raise RuntimeError(f"Cannot find model state_dict in checkpoint keys: {ckpt.keys()}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert os.path.exists(IK_PT_PATH), f"IK pt not found: {IK_PT_PATH}"

    ckpt = torch.load(IK_PT_PATH, map_location=device)
    print("Checkpoint type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("Checkpoint keys:", ckpt.keys())

    model = NeuralIKNet().to(device)

    state_dict = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    # 随便构造一个目标 command
    # [l, p, y, roll, pitch, yaw]
    test_cmd = torch.tensor(
        [[0.40, 0.20, 0.00, 0.10, 0.50, 0.00]],
        dtype=torch.float32,
        device=device
    )

    with torch.inference_mode():
        q_pred = model(test_cmd)

    print("Input command shape:", test_cmd.shape)
    print("Output q shape:", q_pred.shape)
    print("Output q:", q_pred.detach().cpu().numpy())

    print("Has NaN:", torch.isnan(q_pred).any().item())
    print("q min:", q_pred.min().item())
    print("q max:", q_pred.max().item())

    assert q_pred.shape[-1] == 6, f"Expected output dim 6, got {q_pred.shape[-1]}"
    assert not torch.isnan(q_pred).any(), "IK output contains NaN!"

    print("[OK] Neural IK model loads and runs.")


if __name__ == "__main__":
    main()