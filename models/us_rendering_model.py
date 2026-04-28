import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from math import pi
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.measure import label as skimage_label, regionprops

# Default Parameters from: https://github.com/Blito/burgercpp/blob/master/examples/ircad11/liver.scene , labels 8, 9 and 12 approximated from other labels
# Define properties for each category (acoustic_imped, attenuation, mu_0, mu_1, sigma_0)
#               [imped,     att,    mu_0,   mu_1,   sigma_0]
background =    [0.0004,    1.64,   0.78,   0.56,   0.1]
soft_tissue =   [1.38,      0.63,   0.5,    0.5,    0.0]
bone =          [7.8,       5.0,    0.78,   0.56,   0.1]
lung =          [1.61,      0.18,   0.001,  0.0,    0.01]
blood =         [1.61,      0.18,   0.001,  0.01,   0.001]
vessel1 =       [1.99,      1.09,   0.6,    0.6,    0.2]
vessel2 =       [1.99,      1.09,   0.1,    0.1,    0.2]
fat =           [1.65,      0.7,    0.4,    0.8,    0.14]
liver =         [0.3,       0.54,   0.3,    0.2,    0.0]
muscle =        [1.63, 0.54, 0.45, 0.64, 0.1]
galbladder =    [1.62, 1.0, 0.4, 0.6, 0.3]

# Set True to use acoustic values tuned from the vessel task (merged from Ultrasound-Vascular-Liver-Vision).
optimized_parameters = False
if optimized_parameters:
    background =    [0.05437, 0.68123, 0.48596, 1.00022, -0.0006]
    soft_tissue =   [1.66314, 0.60619, 0.47503, 0.73303, -0.00048]
    bone =          [7.92087, 4.51606, 0.66381, 0.4824,  0.05988]
    lung =          [0.12937, 1.43013, 0.62888, 0.50351, 0.01646]
    blood =         [1.2318,  0.14614, -0.00096, -0.00032, -0.00038]
    fat =           [1.44778, 0.86469, 0.39067, 0.49459, -0.0006]
    liver =         [1.98314, 0.70747, 0.4221,  0.97696, -0.00046]
    vessel1 =       [1.5,     0.7,     0.5,     0.5,     0.1]
    vessel2 =       [1.4,     0.8,     0.2,     0.2,     0.1]

# Per-dataset label → material-parameter mappings.
# Pick one at runtime via hparams.dataset_kind ("phantom" | "human").
DATASET_PRESETS = {
    "phantom": {
        "label_parameters": {
            1: background, 2: soft_tissue, 3: bone, 4: lung,
            5: soft_tissue, 6: soft_tissue, 7: soft_tissue, 8: blood,
            9: liver, 11: blood, 12: blood, 13: blood, 14: blood,
            15: blood, 16: blood, 17: blood, 18: blood, 19: blood,
            20: blood, 21: vessel1, 22: blood, 23: vessel2,
            24: soft_tissue, 25: fat,
        },
        "labels": [
            "background", "soft tissue", "bone", "lung", "stomach", "pancreas",
            "spleen", "gallbladder", "liver", "MPV", "LPV", "PV-II", "PV-III",
            "PV-IV", "RPV", "PV-V", "PV-VI", "PV-VII", "PV-VIII",
            "Portal vein wall", "HV", "Hepatic vein wall",
            "Inferior Vena Cava", "Tumor",
        ],
    },
    "human": {
        "label_parameters": {
            0: background, 1: background, 2: soft_tissue, 3: liver,
            4: blood, 5: blood, 6: blood, 7: blood, 8: blood, 9: blood,
            10: blood, 11: blood, 12: blood, 13: blood, 14: blood,
            15: blood, 16: blood, 17: vessel1,
        },
        "labels": [
            "Background", "Background", "Soft tissue", "Liver", "IVC",
            "RHV", "MHV", "LHV", "MPV", "Ant-RPV", "Post-RPV", "RPV",
            "LPV", "II", "III", "IVa", "IVb", "Portal vein wall",
        ],
    },
}


def _build_preset_tensors(dataset_name):
    preset = DATASET_PRESETS[dataset_name]
    label_parameters = preset["label_parameters"]
    keys = sorted(label_parameters.keys())
    make = lambda idx: torch.tensor(
        [label_parameters[k][idx] for k in keys], requires_grad=True
    ).to('cuda')
    mapping_keys = torch.tensor(keys, dtype=torch.long, device='cuda')
    return {
        "acoustic_imped": make(0),
        "attenuation":    make(1),
        "mu_0":           make(2),
        "mu_1":           make(3),
        "sigma_0":        make(4),
        "mapping_keys":   mapping_keys,
        "labels":         preset["labels"],
    }

alpha_coeff_boundary_map = 0.1
beta_coeff_scattering = 10  #100 approximates it closer
TGC = 8
CLAMP_VALS = True


def gaussian_kernel(size: int, mean: float, std: float):
    d1 = torch.distributions.Normal(mean, std)
    d2 = torch.distributions.Normal(mean, std*3)
    vals_x = d1.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()
    vals_y = d2.log_prob(torch.arange(-size, size+1, dtype=torch.float32)).exp()

    gauss_kernel = torch.einsum('i,j->ij', vals_x, vals_y)
    
    return gauss_kernel / torch.sum(gauss_kernel).reshape(1, 1)

g_kernel = gaussian_kernel(3, 0., 0.5)
g_kernel = torch.tensor(g_kernel[None, None, :, :], dtype=torch.float32).to(device='cuda')


def add_ultrasound_noise(image, speckle_std=0.1, gaussian_std=0.05):
    """Simulate ultrasound speckle + Gaussian blur on a grayscale image tensor in [0, 1]."""
    if image.max() > 1:
        image = image / 255.0

    speckle = torch.normal(1, speckle_std, image.shape, device=image.device)
    noisy_image = image * speckle

    noisy_image = noisy_image.unsqueeze(0).unsqueeze(0)
    noisy_image = TF.gaussian_blur(noisy_image, kernel_size=3, sigma=gaussian_std)
    noisy_image = noisy_image.squeeze(0).squeeze(0)

    return torch.clamp(noisy_image, 0, 1)


class UltrasoundRendering(torch.nn.Module):
    def __init__(self, params, default_param=False):
        super(UltrasoundRendering, self).__init__()
        self.params = params

        dataset_name = getattr(params, "dataset_kind", "phantom")
        if dataset_name not in DATASET_PRESETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Expected one of "
                f"{list(DATASET_PRESETS.keys())}."
            )
        preset = _build_preset_tensors(dataset_name)
        self.mapping_keys = preset["mapping_keys"]
        self.labels = preset["labels"]

        if default_param:
            self.acoustic_impedance_dict = preset["acoustic_imped"].detach().clone()
            self.attenuation_dict        = preset["attenuation"].detach().clone()
            self.mu_0_dict               = preset["mu_0"].detach().clone()
            self.mu_1_dict               = preset["mu_1"].detach().clone()
            self.sigma_0_dict            = preset["sigma_0"].detach().clone()
        else:
            self.acoustic_impedance_dict = torch.nn.Parameter(preset["acoustic_imped"])
            self.attenuation_dict        = torch.nn.Parameter(preset["attenuation"])
            self.mu_0_dict               = torch.nn.Parameter(preset["mu_0"])
            self.mu_1_dict               = torch.nn.Parameter(preset["mu_1"])
            self.sigma_0_dict            = torch.nn.Parameter(preset["sigma_0"])

        self.attenuation_medium_map, self.acoustic_imped_map, self.sigma_0_map, self.mu_1_map, self.mu_0_map  = ([] for i in range(5))

    def map_dict_to_array(self, dictionary, arr):
        mapping_keys = self.mapping_keys
        keys = torch.unique(arr)

        index = torch.where(mapping_keys[None, :] == keys[:, None])[1]
        values = torch.gather(dictionary, dim=0, index=index)
        values = values.to(device='cuda')
        # values.register_hook(lambda grad: print(grad))    # check the gradient during training

        mapping = torch.zeros(keys.max().item() + 1).to(device='cuda')
        mapping[keys] = values
        return mapping[arr]
    

    def plot_fig(self, fig, fig_name, grayscale):
        save_dir='results_test/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        plt.clf()

        if torch.is_tensor(fig):
            fig = fig.cpu().detach().numpy()

        if grayscale:
            plt.imshow(fig, cmap='gray', vmin=0, vmax=1, interpolation='none', norm=None)
        else:
            plt.imshow(fig, interpolation='none', norm=None)
        plt.axis('off')
        plt.savefig(save_dir + fig_name + '.png', bbox_inches='tight',transparent=True, pad_inches=0)


    def clamp_map_ranges(self):
        self.attenuation_medium_map = torch.clamp(self.attenuation_medium_map, 0, 10)
        self.acoustic_imped_map = torch.clamp(self.acoustic_imped_map, 0, 10)
        self.sigma_0_map = torch.clamp(self.sigma_0_map, 0, 1)
        self.mu_1_map = torch.clamp(self.mu_1_map, 0, 1)
        self.mu_0_map = torch.clamp(self.mu_0_map, 0, 1)


    def rendering(self, H, W, z_vals=None, refl_map=None, boundary_map=None):
        
        dists = torch.abs(z_vals[..., :-1, None] - z_vals[..., 1:, None])     # dists.shape=(W, H-1, 1)
        dists = dists.squeeze(-1)                                             # dists.shape=(W, H-1)
        dists = torch.cat([dists, dists[:, -1, None]], dim=-1)                # dists.shape=(W, H)

        attenuation = torch.exp(-self.attenuation_medium_map * dists)
        attenuation_total = torch.cumprod(attenuation, dim=1, dtype=torch.float32, out=None)

        gain_coeffs = np.linspace(1, TGC, attenuation_total.shape[1])
        gain_coeffs = np.tile(gain_coeffs, (attenuation_total.shape[0], 1))
        gain_coeffs = torch.tensor(gain_coeffs).to(device='cuda') 
        attenuation_total = attenuation_total * gain_coeffs     # apply TGC

        reflection_total = torch.cumprod(1. - refl_map * boundary_map, dim=1, dtype=torch.float32, out=None) 
        reflection_total = reflection_total.squeeze(-1) 
        reflection_total_plot = torch.log(reflection_total + torch.finfo(torch.float32).eps)

        texture_noise = torch.randn(H, W, dtype=torch.float32).to(device='cuda')
        scattering_probability = torch.randn(H, W, dtype=torch.float32).to(device='cuda') 

        scattering_zero = torch.zeros(H, W, dtype=torch.float32).to(device='cuda')

        z = self.mu_1_map - scattering_probability
        sigmoid_map = torch.sigmoid(beta_coeff_scattering * z)

        # approximating  Eq. (4) to be differentiable:
        # where(scattering_probability <= mu_1_map, 
        #                     texture_noise * sigma_0_map + mu_0_map, 
        #                     scattering_zero)
        scatterers_map =  (sigmoid_map) * (texture_noise * self.sigma_0_map + self.mu_0_map) + (1 -sigmoid_map) * scattering_zero   # Eq. (6)

        psf_scatter_conv = torch.nn.functional.conv2d(input=scatterers_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")
        psf_scatter_conv = psf_scatter_conv.squeeze()

        b = attenuation_total * psf_scatter_conv    # Eq. (3)

        border_convolution = torch.nn.functional.conv2d(input=boundary_map[None, None, :, :], weight=g_kernel, stride=1, padding="same")
        border_convolution = border_convolution.squeeze()

        r = attenuation_total * reflection_total * refl_map * border_convolution # Eq. (2)
        
        intensity_map = b + r   # Eq. (1)
        intensity_map = intensity_map.squeeze() 
        intensity_map = torch.clamp(intensity_map, 0, 1)

        return intensity_map, attenuation_total, reflection_total_plot, scatterers_map, scattering_probability, border_convolution, texture_noise, b, r


    def render_rays(self, W, H):
        N_rays = W 
        t_vals = torch.linspace(0., 1., H).to(device='cuda')   # 0-1 linearly spaced, shape H
        z_vals = t_vals.unsqueeze(0).expand(N_rays , -1) * 4 

        return z_vals 

    # warp the linear US image to approximate US image from curvilinear US probe 
    def warp_img(self, inputImage):
        resultWidth = 290
        resultHeight = 200
        centerX = resultWidth/2
        centerY = -50
        maxAngle = np.pi * 35 / 180
        minAngle = -maxAngle
        minRadius = 61
        maxRadius = 250
        
        h, w = inputImage.squeeze().shape

        # Create x and y grids
        x = torch.arange(resultWidth).float() - centerX
        y = torch.arange(resultHeight).float() - centerY
        xx, yy = torch.meshgrid(x, y)

        # Calculate angle and radius
        angle = torch.atan2(xx, yy)
        radius = torch.sqrt(xx ** 2 + yy ** 2)

        # Create masks for angle and radius
        angle_mask = (angle > minAngle) & (angle < maxAngle)
        radius_mask = (radius > minRadius) & (radius < maxRadius)

        # Calculate original column and row
        origCol = (angle - minAngle) / (maxAngle - minAngle) * w
        origRow = (radius - minRadius) / (maxRadius - minRadius) * h

        # Reshape input image to be a batch of 1 image
        inputImage = inputImage.float().unsqueeze(0).unsqueeze(0)

        # Scale original column and row to be in the range [-1, 1]
        origCol = origCol / (w - 1) * 2 - 1
        origRow = origRow / (h - 1) * 2 - 1

        # Transpose input image to have channels first
        inputImage = inputImage.permute(0, 1, 3, 2)

        # Use grid_sample to interpolate
        grid = torch.stack([origCol, origRow], dim=-1).unsqueeze(0).to('cuda')
        resultImage = F.grid_sample(inputImage, grid, mode='bilinear', align_corners=True)

        # Apply masks and set values outside of mask to 0
        resultImage[~(angle_mask.unsqueeze(0).unsqueeze(0) & radius_mask.unsqueeze(0).unsqueeze(0))] = 0.0
        resultImage_resized = transforms.Resize((256,256))(resultImage).float().squeeze()

        return resultImage_resized
    
    # warp the label image to approximate the label image from curvilinear US probe
    def warp_label(self, inputLabel):
        resultWidth = 290
        resultHeight = 200
        centerX = resultWidth / 2
        centerY = -50
        maxAngle = np.pi * 35 / 180
        minAngle = -maxAngle
        minRadius = 61
        maxRadius = 250

        h, w = inputLabel.squeeze().shape

        # Create x and y grids
        x = torch.arange(resultWidth).float() - centerX
        y = torch.arange(resultHeight).float() - centerY
        xx, yy = torch.meshgrid(x, y)

        # Calculate angle and radius
        angle = torch.atan2(xx, yy)
        radius = torch.sqrt(xx ** 2 + yy ** 2)

        # Create masks for angle and radius
        angle_mask = (angle > minAngle) & (angle < maxAngle)
        radius_mask = (radius > minRadius) & (radius < maxRadius)

        # Calculate original column and row
        origCol = (angle - minAngle) / (maxAngle - minAngle) * w
        origRow = (radius - minRadius) / (maxRadius - minRadius) * h

        # Reshape input label to be a batch of 1 label
        inputLabel = inputLabel.float().unsqueeze(0).unsqueeze(0)

        # Scale original column and row to be in the range [-1, 1]
        origCol = origCol / (w - 1) * 2 - 1
        origRow = origRow / (h - 1) * 2 - 1

        # Transpose input label to have channels first
        inputLabel = inputLabel.permute(0, 1, 3, 2)

        # Use grid_sample to interpolate
        grid = torch.stack([origCol, origRow], dim=-1).unsqueeze(0).to('cuda')
        resultLabel = F.grid_sample(inputLabel, grid, mode='nearest', align_corners=True)

        # Apply masks and set values outside of mask to 0
        resultLabel[~(angle_mask.unsqueeze(0).unsqueeze(0) & radius_mask.unsqueeze(0).unsqueeze(0))] = 0.0

        # Convert resultLabel to numpy array for processing
        resultLabel_resized = transforms.Resize((256, 256))(resultLabel).float().squeeze()
        resultLabel_np = resultLabel_resized.cpu().numpy()

        # Create a combined mask of all labels
        combined_mask = (resultLabel_np > 0).astype(int)

        # Apply dilatation
        for label in range(4, 0, -1):
            binary_label = (resultLabel_np == label).astype(int)
            # Use a larger 7x7 kernel for dilation
            dilated_label = binary_dilation(binary_label, structure=np.ones((6, 6))).astype(int)
            resultLabel_np[dilated_label == 1] = label

        # Apply the combined mask
        resultLabel_np *= combined_mask  # Mask all labels based on the combined mask

        # Convert back to torch tensor
        resultLabel_resized = torch.from_numpy(resultLabel_np).float()

        return resultLabel_resized

    def forward(self, ct_slice):
        if self.params.debug: self.plot_fig(ct_slice, "ct_slice", False)        

        #init tissue maps
        #generate 2D acousttic_imped map
        self.acoustic_imped_map = self.map_dict_to_array(self.acoustic_impedance_dict, ct_slice)#.astype('int64'))

        #generate 2D attenuation map
        self.attenuation_medium_map = self.map_dict_to_array(self.attenuation_dict, ct_slice)

        if self.params.debug:   
            self.plot_fig(self.acoustic_imped_map, "acoustic_imped_map", False)
            self.plot_fig(self.attenuation_medium_map, "attenuation_medium_map", False)

        self.mu_0_map = self.map_dict_to_array(self.mu_0_dict, ct_slice)

        self.mu_1_map = self.map_dict_to_array(self.mu_1_dict, ct_slice)

        self.sigma_0_map = self.map_dict_to_array(self.sigma_0_dict, ct_slice)

        self.acoustic_imped_map = torch.rot90(self.acoustic_imped_map, 1, [0, 1])
        diff_arr = torch.diff(self.acoustic_imped_map, dim=0)

        diff_arr = torch.cat((torch.zeros(diff_arr.shape[1], dtype=torch.float32).unsqueeze(0).to(device='cuda'), diff_arr))

        boundary_map =  -torch.exp(-(diff_arr**2)/alpha_coeff_boundary_map) + 1

        boundary_map = torch.rot90(boundary_map, 3, [0, 1])

        if self.params.debug:
           self.plot_fig(diff_arr, "diff_arr", False)
           self.plot_fig(boundary_map, "boundary_map", True)

        shifted_arr = torch.roll(self.acoustic_imped_map, -1, dims=0)
        shifted_arr[-1:] = 0

        sum_arr = self.acoustic_imped_map + shifted_arr
        sum_arr[sum_arr == 0] = 1
        div = diff_arr / sum_arr

        refl_map = div ** 2
        refl_map = torch.sigmoid(refl_map)      # 1 / (1 + (-refl_map).exp())
        refl_map = torch.rot90(refl_map, 3, [0, 1])

        if self.params.debug: self.plot_fig(refl_map, "refl_map", True)

        z_vals = self.render_rays(ct_slice.shape[0], ct_slice.shape[1])

        if CLAMP_VALS:
            self.clamp_map_ranges()

        ret_list = self.rendering(ct_slice.shape[0], ct_slice.shape[1], z_vals=z_vals, refl_map=refl_map, boundary_map=boundary_map)

        intensity_map  = ret_list[0]

        if self.params.debug:  
            self.plot_fig(intensity_map, "intensity_map", True)

            result_list = ["intensity_map", "attenuation_total", "reflection_total", 
                            "scatters_map", "scattering_probability", "border_convolution", 
                            "texture_noise", "b", "r"]

            for k in range(len(ret_list)):
                result_np = ret_list[k]
                if torch.is_tensor(result_np):
                    result_np = result_np.detach().cpu().numpy()
                        
                if k==2:
                    self.plot_fig(result_np, result_list[k], False)
                else:
                    self.plot_fig(result_np, result_list[k], True)
                # print(result_list[k], ", ", result_np.shape)

        intensity_map_masked = self.warp_img(intensity_map)
        intensity_map_masked = torch.rot90(intensity_map_masked, 3)
        
        if self.params.debug:  self.plot_fig(intensity_map_masked, "intensity_map_masked", True)

        return intensity_map_masked

