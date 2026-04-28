import numpy as np
import os
from os.path import join
import nibabel as nib
import torch

def TensorFieldInVolume(field, ROI, volume_shape, DOFs=None):
    DOFs = DOFs if DOFs is not None else slice(None)
    shape = field.shape[:-4] + (3,) + tuple(volume_shape)
    _ = torch.zeros(shape, dtype=field.dtype, device=field.device)
    _[..., DOFs, ROI[0] : ROI[3], ROI[1] : ROI[4], ROI[2] : ROI[5]] = field
    return _


def upsample_disp_field_to_roi(
    field,
    upsample_shape,
    roi=None,
    DOFs=None,
    device=None,
    display=False,
    adapt_from_domain=None,
):
    """
    :param field: (N,nDOFs,I,J,K) or (nDOFs,I,J,K) size with N the batch size, nDOFs between 1 and 3.
    :param upsample_shape: (I,J,K) output size of the field, with padding if roi is not None
    :param roi: (i0,j0,k0,i1,j1,k1) defines both the shape the field will be upsampled to and the padding around it.
    :param DOFs: The degrees of freedom of the field. If None, the field is should have 3 components.
    :param device: The device on which the output field will be.
    :param display: If True, display the shape and min/max of the output field.
    :param adapt_from_domain: The size the original field was defined on.
    e.g. original field defined on an roi of a certain shape, and upsampled to a roi 2 times smaller.
    default to None means the original field was defined on roi (or upsample_shape if roi is None).
    """
    if roi is None:
        roi = tuple([0] * len(upsample_shape)) + upsample_shape
    while len(field.shape) < 5:
        field = field.unsqueeze(0)
    if field.shape[-4] != 3 and DOFs is None:
        raise ValueError(
            f"Field shape {field.shape} is not 3D and DOFs is not provided"
        )
    # First upsample the field to match the size of the roi
    roi_shape = tuple(roi[i + 3] - roi[i] for i in range(len(roi) // 2))
    field_in_volume = torch.nn.functional.interpolate(
        field, size=roi_shape, mode="trilinear", align_corners=True
    )
    # Then add padding around the field to match the upsample_shape
    field_in_volume = TensorFieldInVolume(
        field=field_in_volume, ROI=roi, volume_shape=upsample_shape, DOFs=DOFs
    ).to(device)
    if adapt_from_domain is not None:
        scale = torch.as_tensor(
            adapt_from_domain, dtype=field.dtype, device=field.device
        ) / torch.as_tensor(roi_shape, dtype=field.dtype, device=field.device)
        field_in_volume = field_in_volume * scale[None, :, None, None, None]
    if display:
        print(
            f"[upsample_field_to_roi] Displacement field upscaled to {field_in_volume.shape} "
            f"min/max: {field_in_volume.min()}, {field_in_volume.max()}"
        )  # 400ms
    return field_in_volume


def save_new_vol(new_vol, affine, header, fname):
    nv = nib.Nifti1Image(new_vol, affine, header)
    nib.save(nv, fname)

def unsqueeze_as(a, b, dim=0):
    n_unsqueeze = len(b.shape) - len(a.shape)
    if dim == 0:
        return a[(None,) * n_unsqueeze + (...,)]
    elif dim == -1:
        return a[(...,) + (None,) * n_unsqueeze]
    else:
        raise ValueError
    

class DisplacementFieldMapperND:
    def __init__(
        self, dims_number, img=None, img_shape=None, device=None, dtype=torch.float32
    ):
        if dims_number > 3:
            raise NotImplementedError("Only 1D, 2D and 3D are supported")

        self.dims_number = dims_number
        self.device = device
        self.dtype = dtype
        self.img = None

        if img is not None:
            self.shape = img.shape[-dims_number:]
            if isinstance(img, torch.Tensor):
                self.img = img.to(device=self.device, dtype=dtype)
            else:
                self.img = torch.tensor(img, device=self.device, dtype=dtype)
        elif img_shape is not None:
            self.shape = img_shape[-dims_number:]
        else:
            self.shape = None

        if self.shape is not None:
            self.disp_coordinates = torch.stack(
                torch.meshgrid(
                    *[
                        torch.linspace(-1, 1, s, device=self.device, dtype=self.dtype)
                        for s in self.shape
                    ],
                    indexing="ij",
                )
            )[None, ...]
            self.scale = ((torch.tensor(self.shape, device=self.device) - 1) * 0.5)[
                None, ...
            ]
            self.scale = unsqueeze_as(self.scale, self.disp_coordinates, dim=-1)

    def __call__(self, disp, img=None, mode="bilinear", *args, **kwargs):
        return self.apply_grid_sample(disp, img, mode, *args, **kwargs)

    def _permute_dims_reverse(self, tensor, dims_number):
        return torch.permute(
            tensor,
            (
                *range(0, len(tensor.shape[:-dims_number])),
                *range(len(tensor.shape) - 1, len(tensor.shape[:-dims_number]) - 1, -1),
            ),
        )

    def map_coordinates(
        self,
        coords,
        img,
        mode,
        padding_mode="border",
        align_corners=True,
        *args,
        **kwargs,
    ):
        """
        Deforms images with coordinates.

        :param coords: (N,D,*dims) size with N the batch size
        :param img: (N,*dims) or (*dims) size with N the batch size
        :param mode: 'bilinear' or 'nearest' or 'bicubic'
        :param padding_mode: 'zeros' or 'border' or 'reflection'
        :param align_corners: True or False
        """
        if len(img.shape) > self.dims_number + 1:
            raise ValueError

        img = self._permute_dims_reverse(img, self.dims_number)
        batched_img = len(img.shape) == self.dims_number + 1

        if not batched_img:
            img = img[None, ...]
        img = img[:, None]

        img = torch.nn.functional.grid_sample(
            img,
            coords,
            mode=mode,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )

        return self._permute_dims_reverse(img, self.dims_number)

    def map_batch_coordinates(
        self,
        coords,
        img,
        mode,
        padding_mode="border",
        align_corners=True,
        *args,
        **kwargs,
    ):
        """
        Maps a batch of coordinates to a single image to obtain a batch of *different* deformed images.

        :param coords: (N,D,*dims) or (D,*dims) size with N the batch size
        :param img: (C,*dims) or (*dims) size with C the channel size
        :param mode: 'bilinear' or 'nearest' or 'bicubic'
        :param padding_mode: 'zeros' or 'border' or 'reflection'
        :param align_corners: True or False
        """
        if len(img.shape) > self.dims_number + 1:
            raise ValueError(
                "Cannot map batch coordinates to a a batch of images."
                "Please use map_coordinates instead."
            )
        img = self._permute_dims_reverse(img, self.dims_number)
        # Adding 1 dim if img shape is (C,*dims) or 2 dims if (*dims)
        add_dims = (self.dims_number + 2) - len(img.shape)
        img = img.view((1,) * add_dims + img.shape)
        # Adding a batch dim if coords shape is (D,*dims)
        if len(coords.shape) < self.dims_number + 2:
            coords = coords.view((1,) + coords.shape)
        # Expanding the batch dim of the image to the number of disps
        same_n = img.shape[0] == coords.shape[0]
        img = img.expand(
            ([coords.shape[0], -1][same_n],) + (-1,) * (len(img.shape) - 1)
        )  # If N of disp != N of img, expand img
        coords = coords.permute(0, *range(len(coords.shape) - 1, 0, -1))
        nimg = torch.nn.functional.grid_sample(
            img,
            coords,
            mode=mode,
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        #                        Keep the first dimensions as they are
        return nimg.permute(
            (
                *range(0, len(nimg.shape[: -self.dims_number])),
                #                        Reverse the last <dims_number> dimensions
                *range(
                    len(nimg.shape) - 1, len(nimg.shape[: -self.dims_number]) - 1, -1
                ),
            )
        )

    def apply_grid_sample(
        self, disp, img=None, mode="bilinear", padding_mode="border", *args, **kwargs
    ):
        img = self.img if img is None else img

        disp = disp[
            (None,) * (len(self.disp_coordinates.shape) - len(disp.shape)) + (...,)
        ]
        disp_grid = (
            self.disp_coordinates.expand(*disp.shape)
            + disp.to(self.disp_coordinates) / self.scale
        ).permute(0, *range(1 + self.dims_number, 0, -1))
        return self.map_coordinates(
            coords=disp_grid,
            img=img,
            mode=mode,
            padding_mode=padding_mode,
            *args,
            **kwargs,
        )

    def img_coordinates(self):
        return (self.disp_coordinates + 1) * self.scale.squeeze(0)

    def to(self, device, dtype=None):
        attrs = ["disp_coordinates", "scale", "img"]
        for attr in attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        self.device = device
        if dtype is not None:
            for attr in attrs:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    setattr(self, attr, getattr(self, attr).to(dtype))
            self.dtype = dtype
        return self


class DisplacementFieldMapper3D(DisplacementFieldMapperND):
    def __init__(self, img=None, img_shape=None, device=None, dtype=torch.float32):
        super(DisplacementFieldMapper3D, self).__init__(
            dims_number=3, img=img, img_shape=img_shape, device=device, dtype=dtype
        )


class DisplacementFieldMapper2D(DisplacementFieldMapperND):
    def __init__(self, img=None, img_shape=None, device=None, dtype=torch.float32):
        super(DisplacementFieldMapper2D, self).__init__(
            dims_number=2, img=img, img_shape=img_shape, device=device, dtype=dtype
        )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)


def run_deform(dataset="human", n_fields=1000, device="cuda"):
    data_dir = os.path.join(BASE_DIR, "deformation_fields")
    if dataset == "human":
        labelmap_file = os.path.join(
            REPO_ROOT,
            "datasets/CT_Liver_labelmaps_phantom/CT001/ct_labelmap/labels-label.nii.gz",
        )
        out_dir = os.path.join(BASE_DIR, "output_human")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    os.makedirs(out_dir, exist_ok=True)

    labelmap = nib.load(labelmap_file)
    labelmap_data = labelmap.get_fdata()
    labelmap_dtype = labelmap_data.dtype
    labelmap_affine = labelmap.affine
    labelmap_header = labelmap.header

    mapper_label = DisplacementFieldMapper3D(labelmap_data, device=device)

    for i in range(n_fields):
        field_file = join(data_dir, f"field_{i}.npy")
        field = torch.as_tensor(np.load(field_file)) * 5
        field = upsample_disp_field_to_roi(field, labelmap.shape)

        deformed_labelmap = (
            mapper_label(field, mode="nearest")
            .squeeze()
            .cpu()
            .numpy()
            .astype(labelmap_dtype)
            .round()
        )

        labelmap_out_path = join(out_dir, f"labelmap_deformed_{i:03d}.nii.gz")
        save_new_vol(deformed_labelmap, labelmap_affine, labelmap_header, labelmap_out_path)

        print(f"Deformed labelmap saved for field_{i}")

    print(f"Successfully generated {n_fields} deformed labelmaps in {out_dir}")


if __name__ == "__main__":
    run_deform(dataset="human")
