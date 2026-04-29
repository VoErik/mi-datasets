import numbers
import warnings

from abc import (
    ABC, 
    abstractmethod
)
from typing import (
    Any, 
    Dict, 
    List,
    Sequence, 
    Tuple,
    Union
)
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop


class TrackedTransform(ABC):
    """Base class for all Mechanistic Interpretability transforms."""
    
    @abstractmethod
    def __call__(self, x: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Applies the transform and returns the transformed data alongside 
        the exact parameters used (the tracking state).
        """
        pass

    @abstractmethod
    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """
        Projects a tensor (e.g., an activation map) back through the transform 
        using the specific parameters logged during the forward pass.
        """
        pass

class TrackedCompose:
    """Chains transforms, natively supporting both MI Tracked and standard Torchvision transforms."""
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms
        self._validate_pipeline()

    def _validate_pipeline(self) -> None:
        """Scans for untracked spatial transforms and suggests framework alternatives."""
        spatial_keywords = ["Crop", "Resize", "Flip", "Rotate", "Pad", "Affine", "Perspective"]
        
        available_tracked = {name for name in globals() if name.startswith("Tracked")}
        
        for t in self.transforms:
            if not hasattr(t, "inverse"):
                name = t.__class__.__name__
                expected_tracked_name = f"Tracked{name}"
                
                if expected_tracked_name in available_tracked:
                    warnings.warn(
                        f"\n[MI WARNING] Standard '{name}' detected.\n"
                        f"The framework has a built-in tracked alternative for this. "
                        f"Please import and use '{expected_tracked_name}' instead to maintain spatial inversion.",
                        UserWarning
                    )
                elif any(kw in name for kw in spatial_keywords):
                    warnings.warn(
                        f"\n[MI WARNING] Untracked spatial transform detected: '{name}'\n"
                        f"Standard spatial transforms break the geometry of the pipeline. "
                        f"Calling .inverse() on this pipeline will result in misaligned activation maps. "
                        f"Please write a {expected_tracked_name} class if you need spatial inversion.",
                        UserWarning
                    )

    def __call__(self, x: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        tracking_history = []
        for t in self.transforms:
            result = t(x)
            
            if isinstance(result, tuple) and len(result) == 2:
                x, params = result
            else:
                x = result
                params = {"tracked": False} 
                
            tracking_history.append({
                "name": t.__class__.__name__,
                "params": params
            })
        return x, tracking_history

    def inverse(self, x: torch.Tensor, tracking_history: List[Dict[str, Any]]) -> torch.Tensor:
        """Runs the inverse transformations in reverse order."""
        for t, track in zip(reversed(self.transforms), reversed(tracking_history)):
            assert t.__class__.__name__ == track["name"], "Transform mismatch during inversion."
            
            if hasattr(t, "inverse"):
                x = t.inverse(x, track["params"])
            elif track["params"].get("tracked") is False:
                pass
                
        return x

class TrackedToTensor(TrackedTransform):
    def __call__(self, img: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        tensor = F.to_tensor(img)
        return tensor, {}

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """
        Converting to a tensor alters the channel dimension order (HWC -> CHW) and scales values, 
        but it does not alter spatial geometry. For inverse spatial mapping, we just pass the tensor through.
        """
        return x

class TrackedNormalize(TrackedTransform):
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        normalized = F.normalize(tensor, self.mean, self.std)
        params = {"mean": self.mean, "std": self.std}
        return normalized, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """
        Un-normalizes the tensor. Useful for visualizing the exact inputs 
        the network received as raw RGB images.
        """
        mean = torch.tensor(params["mean"], device=x.device).view(-1, 1, 1)
        std = torch.tensor(params["std"], device=x.device).view(-1, 1, 1)
        return x * std + mean

##############################################
################### RESIZES ##################
##############################################

class TrackedResize(TrackedTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, orig_h, orig_w = F.get_dimensions(img)
        resized_img = F.resize(img, self.size)
        
        params = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": self.size[0],
            "new_w": self.size[1],
            "scale_factor_h": self.size[0] / orig_h,
            "scale_factor_w": self.size[1] / orig_w
        }
        return resized_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return F.resize(x, [params["orig_h"], params["orig_w"]])

##############################################
################### CROPS ####################
##############################################

class TrackedCenterCrop(TrackedTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, h, w = F.get_dimensions(img)
        th, tw = self.size
        
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        
        cropped_img = F.crop(img, i, j, th, tw)
        
        params = {
            "crop_top": i, 
            "crop_left": j, 
            "crop_h": th, 
            "crop_w": tw,
            "orig_h": h,
            "orig_w": w
        }
        return cropped_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        pad_left = params["crop_left"]
        pad_right = params["orig_w"] - (params["crop_left"] + params["crop_w"])
        pad_top = params["crop_top"]
        pad_bottom = params["orig_h"] - (params["crop_top"] + params["crop_h"])
        
        return F.pad(x, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

class TrackedRandomCrop(TrackedTransform):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, h, w = F.get_dimensions(img)
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
            
        i = int(torch.randint(0, h - th + 1, size=(1,)).item())
        j = int(torch.randint(0, w - tw + 1, size=(1,)).item())
        
        cropped_img = F.crop(img, i, j, th, tw)
        
        params = {
            "crop_top": i, 
            "crop_left": j, 
            "crop_h": th, 
            "crop_w": tw,
            "orig_h": h,
            "orig_w": w
        }
        return cropped_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        pad_left = params["crop_left"]
        pad_right = params["orig_w"] - (params["crop_left"] + params["crop_w"])
        pad_top = params["crop_top"]
        pad_bottom = params["orig_h"] - (params["crop_top"] + params["crop_h"])
        
        return F.pad(x, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

class TrackedRandomResizedCrop(TrackedTransform):
    def __init__(
        self, 
        size: Tuple[int, int], 
        scale: Tuple[float, float] = (0.08, 1.0), 
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, orig_h, orig_w = F.get_dimensions(img)
        
        i, j, h, w = RandomResizedCrop.get_params(img, self.scale, self.ratio)

        cropped_img = F.crop(img, i, j, h, w)
        resized_img = F.resize(cropped_img, self.size)

        params = {
            "crop_top": i,
            "crop_left": j,
            "crop_h": h,
            "crop_w": w,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": self.size[0],
            "new_w": self.size[1]
        }
        return resized_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        x_unresized = F.resize(x, [params["crop_h"], params["crop_w"]])
        
        pad_left = params["crop_left"]
        pad_right = params["orig_w"] - (params["crop_left"] + params["crop_w"])
        pad_top = params["crop_top"]
        pad_bottom = params["orig_h"] - (params["crop_top"] + params["crop_h"])
        
        return F.pad(x_unresized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

##############################################
################# ROTATIONS ##################
##############################################

class TrackedRandomRotation(TrackedTransform):
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, img: Any) -> Tuple[Any, Dict[str, Any]]:
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        
        rotated_img = F.rotate(img, angle)
        
        params = {
            "angle": angle,
            "center": None, # Default is center
            "expand": False
        }
        return rotated_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """
        Rotates the activation map back by -angle.
        Note: Rotation inversion often introduces black borders (zero-padding).
        """
        return F.rotate(x, -params["angle"])

##############################################
#################### FLIPS ###################
##############################################

class TrackedRandomHorizontalFlip(TrackedTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Any) -> Tuple[Any, Dict[str, Any]]:
        was_flipped = float(torch.rand(1).item()) < self.p
        if was_flipped:
            img = F.hflip(img)
            
        params = {"was_flipped": was_flipped}
        return img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        if params["was_flipped"]:
            return F.hflip(x)
        return x

class TrackedRandomVerticalFlip(TrackedTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Any) -> Tuple[Any, Dict[str, Any]]:
        was_flipped = float(torch.rand(1).item()) < self.p
        if was_flipped:
            img = F.vflip(img)
            
        params = {"was_flipped": was_flipped}
        return img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        if params["was_flipped"]:
            return F.vflip(x)
        return x

##############################################
################### PADDING ##################
##############################################

class TrackedPad(TrackedTransform):
    def __init__(self, padding: Union[int, Sequence[int]], fill: int = 0):
        self.padding = padding
        self.fill = fill

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, h, w = F.get_dimensions(img)

        # Standardize padding into [left, top, right, bottom]
        if isinstance(self.padding, numbers.Number):
            pad_left = pad_right = pad_top = pad_bottom = int(self.padding)
        elif len(self.padding) == 2:
            pad_left = pad_right = int(self.padding[0])
            pad_top = pad_bottom = int(self.padding[1])
        else:
            pad_left, pad_top, pad_right, pad_bottom = self.padding

        padded_img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

        params = {
            "pad_left": pad_left,
            "pad_top": pad_top,
            "pad_right": pad_right,
            "pad_bottom": pad_bottom,
            "orig_h": h,
            "orig_w": w
        }
        return padded_img, params

    def inverse(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        # The inverse of padding is a crop starting at (top, left) with the original dimensions
        return F.crop(
            x,
            top=params["pad_top"],
            left=params["pad_left"],
            height=params["orig_h"],
            width=params["orig_w"]
        )
    
# TODO: RandomAffine (requires computing the inverse of the affine matrix)
# TODO: RandomPerspective (requires computing the inverse homography matrix)
# TODO: ElasticTransform (requires solving an inverse optical flow field)