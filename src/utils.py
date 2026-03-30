from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import torch


def dwt_pyramid(image: torch.Tensor, levels: int, wavelet: str = 'db1'):
    dwt = DWTForward(J=levels, wave=wavelet, mode='symmetric').to(image.device)
    LL, details_list = dwt(image)
    pyramid = [LL] + list(reversed(details_list))
    return pyramid

def build_image_from_dwt_pyramid(pyramid, wavelet: str = 'db1'):
    LL = pyramid[0]
    details_list = list(reversed(pyramid[1:]))
    idwt = DWTInverse(wave=wavelet, mode='symmetric').to(LL.device)
    image = idwt((LL, details_list))
    return image

def adaptive_weight(v_ref: torch.Tensor, v_edit: torch.Tensor, base_alpha: float, gamma: float):
    diff = (v_edit - v_ref).pow(2).sum(dim=1, keepdim=True).sqrt()
    diff_min = diff.amin(dim=(2, 3), keepdim=True)
    diff_max = diff.amax(dim=(2, 3), keepdim=True)
    diff_norm = 1 - (diff - diff_min) / (diff_max - diff_min + 1e-8)
    alpha_map = base_alpha * (torch.exp(diff_norm * gamma) - 1)
    return alpha_map

def correct_with_wavelet_guidance(
        v_pred: torch.Tensor, v_ref: torch.Tensor, alpha_map: torch.Tensor,
        wavelet: str = 'db4', levels: int = 2
        ):
    
    dtype = v_pred.dtype
    
    v_pred = v_pred.to(torch.float32)
    v_ref = v_ref.to(torch.float32)

    pred_pyramid = dwt_pyramid(v_pred, levels, wavelet)
    keep_pyramid = dwt_pyramid(v_ref, levels, wavelet)
    
    corrected_pyramid = []
    
    corrected_pyramid.append(pred_pyramid[0])  

    for i in range(1, levels + 1):
        p_pred = pred_pyramid[i]
        p_keep = keep_pyramid[i]
        
        Ht, Wt = p_pred.shape[-2:]

        alpha_map = F.interpolate(alpha_map, size=(Ht, Wt),
                                   mode="bilinear", align_corners=False)
        p_corrected = (1 - alpha_map) * p_pred + alpha_map * p_keep
        corrected_pyramid.append(p_corrected)

    v_corrected = build_image_from_dwt_pyramid(corrected_pyramid, wavelet)
    return v_corrected.to(dtype=dtype)