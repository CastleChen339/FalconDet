import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from ..core import BaseConfig
from ..misc import dist


def remove_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


class BaseSolver:
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg

    def _setup(self):
        cfg = self.cfg
        device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = cfg.model
        eval_spatial_size = cfg.global_cfg["eval_spatial_size"]
        input_tensor = torch.randn(
            1,
            3,
            eval_spatial_size[0],
            eval_spatial_size[1],
            eval_spatial_size[2],
        )
        try:
            from thop import profile

            flops, params = profile(self.model, inputs=(input_tensor,), verbose=False)
            print(f"FLOPs: {flops / 1e9:.2f} G")
            print(f"Params: {params / 1e6:.2f} M")
        except Exception:
            print("FLOPs profiling skipped.")

        if cfg.tuning:
            print(f"Tuning checkpoint from {cfg.tuning}")
            self.load_tuning_state(cfg.tuning)

        self.model = dist.warp_model(
            self.model.to(device),
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters,
        )
        self.criterion = self.to(cfg.criterion, device)
        self.postprocessor = self.to(cfg.postprocessor, device)
        self.ema = self.to(cfg.ema, device)

        self.device = device
        self.last_epoch = cfg.last_epoch

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer
        if self.writer:
            atexit.register(self.writer.close)
            if dist.is_main_process():
                self.writer.add_text("config", repr(cfg), 0)

    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler
        self.train_dataloader = dist.warp_loader(
            self.cfg.train_dataloader,
            shuffle=self.cfg.train_dataloader.shuffle,
        )
        self.val_dataloader = dist.warp_loader(
            self.cfg.val_dataloader,
            shuffle=self.cfg.val_dataloader.shuffle,
        )

        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.load_resume_state(self.cfg.resume)

    def eval(self):
        self._setup()
        self.val_dataloader = dist.warp_loader(
            self.cfg.val_dataloader,
            shuffle=self.cfg.val_dataloader.shuffle,
        )

        if self.cfg.resume:
            print(f"Resume checkpoint from {self.cfg.resume}")
            self.load_resume_state(self.cfg.resume)

    @staticmethod
    def to(module, device):
        return module.to(device) if hasattr(module, "to") else module

    def state_dict(self):
        state = {
            "date": datetime.now().isoformat(),
            "last_epoch": self.last_epoch,
        }
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict"):
                state[key] = dist.de_parallel(value).state_dict()
        return state

    def load_state_dict(self, state):
        if "last_epoch" in state:
            self.last_epoch = state["last_epoch"]
            print("Load last_epoch")

        for key, value in self.__dict__.items():
            if not hasattr(value, "load_state_dict"):
                continue
            if key in state:
                dist.de_parallel(value).load_state_dict(state[key])
                print(f"Load {key}.state_dict")
            elif key == "ema":
                model = getattr(self, "model", None)
                if model is not None:
                    ema = dist.de_parallel(value)
                    model_state = remove_module_prefix(model.state_dict())
                    ema.load_state_dict({"module": model_state})
                    print("Load ema.state_dict from model.state_dict")
            else:
                print(f"Not load {key}.state_dict")

    @staticmethod
    def _clean_profiling_keys(state):
        return {
            key: value
            for key, value in state.items()
            if not key.endswith(("total_ops", "total_params"))
        }

    @staticmethod
    def _load_checkpoint(path: str):
        if path.startswith("http"):
            return torch.hub.load_state_dict_from_url(path, map_location="cpu")
        return torch.load(path, map_location="cpu")

    def load_resume_state(self, path: str):
        state = self._load_checkpoint(path)
        if "model" in state:
            state["model"] = self._clean_profiling_keys(state["model"])
        if "ema" in state and "module" in state["ema"]:
            state["ema"]["module"] = self._clean_profiling_keys(
                state["ema"]["module"]
            )
        self.load_state_dict(state)

    def load_tuning_state(self, path: str):
        state = self._load_checkpoint(path)
        if "ema" in state:
            pretrained = state["ema"]["module"]
        else:
            pretrained = state.get("model", state)

        pretrained = remove_module_prefix(
            self._clean_profiling_keys(pretrained)
        )
        module = dist.de_parallel(self.model)
        matched, info = self._matched_state(module.state_dict(), pretrained)
        module.load_state_dict(matched, strict=False)
        print(f"Load model.state_dict, {info}")

    @staticmethod
    def _matched_state(
        state: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
    ):
        missed = []
        unmatched = []
        matched = {}
        for key, value in state.items():
            if key not in params:
                missed.append(key)
            elif value.shape != params[key].shape:
                unmatched.append(key)
            else:
                matched[key] = params[key]
        return matched, {"missed": missed, "unmatched": unmatched}

    def fit(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError
