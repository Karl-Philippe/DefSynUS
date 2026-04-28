import logging
import os
import sys
import time
import traceback
import contextlib
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin


LOGGER = logging.getLogger(__name__)
HARD_CODED_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULT_VESSEL_LABEL_LEGEND = {
    0: ("BG", (0.0, 0.0, 0.0, 0.0)),
    1: ("MPV", (0.5, 0.0, 0.5, 1.0)),       # #800080
    2: ("LPV", (0.0, 0.0, 1.0, 1.0)),       # #0000FF
    3: ("RPV", (1.0, 0.0, 0.0, 1.0)),       # #FF0000
    4: ("HV", (0.0, 191/255.0, 1.0, 1.0)),  # #00BFFF
}


def _pair(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected 2 values, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _same_padding_2d(kernel_size: Union[int, Sequence[int]], dilation: Union[int, Sequence[int]] = 1) -> Tuple[int, int]:
    kx, ky = _pair(kernel_size)
    dx, dy = _pair(dilation)
    # Matches MONAI same_padding for odd kernels used here (1x1 and 3x3).
    return ((kx - 1) * dx) // 2, ((ky - 1) * dy) // 2


def _stride_minus_kernel_padding_2d(kernel_size: Union[int, Sequence[int]], stride: Union[int, Sequence[int]]) -> Tuple[int, int]:
    kx, ky = _pair(kernel_size)
    sx, sy = _pair(stride)
    return sx - kx, sy - ky


def _build_lite_monai_attention_unet(
    torch_module,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: Sequence[int],
    strides: Sequence[int],
    kernel_size: Union[int, Sequence[int]] = 3,
    up_kernel_size: Union[int, Sequence[int]] = 3,
    dropout: float = 0.0,
):
    if spatial_dims != 2:
        raise ValueError("The lightweight MONAI AttentionUnet fallback only supports spatial_dims=2.")
    if len(channels) < 2:
        raise ValueError("channels must contain at least two levels.")
    if len(strides) < len(channels) - 1:
        raise ValueError("strides length must be at least len(channels)-1.")

    nn = torch_module.nn

    class ADN(nn.Sequential):
        def __init__(
            self,
            ordering: str = "NDA",
            in_channels: int = 1,
            act: Optional[Union[Tuple, str]] = "PRELU",
            norm: Optional[Union[Tuple, str]] = "INSTANCE",
            norm_dim: int = 2,
            dropout: Optional[Union[Tuple, str, float]] = None,
            dropout_dim: Optional[int] = 1,
        ):
            super().__init__()
            if norm_dim != 2:
                raise ValueError("Only 2D norm is supported in this fallback.")

            act_name = act[0] if isinstance(act, tuple) else act
            norm_name = norm[0] if isinstance(norm, tuple) else norm
            drop_value = dropout[0] if isinstance(dropout, tuple) else dropout

            norm_layer = None
            if norm_name is not None:
                norm_str = str(norm_name).lower()
                if "batch" in norm_str:
                    norm_layer = nn.BatchNorm2d(in_channels)
                elif "instance" in norm_str:
                    # MONAI default instance norm here has no affine params/running stats.
                    norm_layer = nn.InstanceNorm2d(in_channels, affine=False, track_running_stats=False)
                else:
                    raise ValueError(f"Unsupported norm '{norm_name}' in fallback AttentionUnet.")

            act_layer = None
            if act_name is not None:
                act_str = str(act_name).lower()
                if act_str == "relu":
                    act_layer = nn.ReLU(inplace=False)
                elif act_str == "prelu":
                    act_layer = nn.PReLU(num_parameters=1)
                else:
                    raise ValueError(f"Unsupported activation '{act_name}' in fallback AttentionUnet.")

            drop_layer = None
            if drop_value not in (None, 0, 0.0, False):
                p = float(drop_value)
                if p > 0:
                    drop_layer = nn.Dropout2d(p=p)

            for symbol in ordering:
                if symbol == "N" and norm_layer is not None:
                    self.add_module("N", norm_layer)
                elif symbol == "D" and drop_layer is not None:
                    self.add_module("D", drop_layer)
                elif symbol == "A" and act_layer is not None:
                    self.add_module("A", act_layer)

    class Convolution(nn.Sequential):
        def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            strides: Union[Sequence[int], int] = 1,
            kernel_size: Union[Sequence[int], int] = 3,
            adn_ordering: str = "NDA",
            act: Optional[Union[Tuple, str]] = "PRELU",
            norm: Optional[Union[Tuple, str]] = "INSTANCE",
            dropout: Optional[Union[Tuple, str, float]] = None,
            dropout_dim: Optional[int] = 1,
            dilation: Union[Sequence[int], int] = 1,
            groups: int = 1,
            bias: bool = True,
            conv_only: bool = False,
            is_transposed: bool = False,
            padding: Optional[Union[Sequence[int], int]] = None,
            output_padding: Optional[Union[Sequence[int], int]] = None,
        ):
            super().__init__()
            if spatial_dims != 2:
                raise ValueError("Fallback Convolution only supports spatial_dims=2.")

            stride2 = _pair(strides)
            kernel2 = _pair(kernel_size)
            dilation2 = _pair(dilation)
            if padding is None:
                padding = _same_padding_2d(kernel2, dilation2)
            padding2 = _pair(padding)

            if is_transposed:
                if output_padding is None:
                    # Matches MONAI stride_minus_kernel_padding(1, strides)
                    output_padding = _stride_minus_kernel_padding_2d(1, stride2)
                output_padding2 = _pair(output_padding)
                conv = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel2,
                    stride=stride2,
                    padding=padding2,
                    output_padding=output_padding2,
                    groups=groups,
                    bias=bias,
                    dilation=dilation2,
                )
            else:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel2,
                    stride=stride2,
                    padding=padding2,
                    dilation=dilation2,
                    groups=groups,
                    bias=bias,
                )

            self.add_module("conv", conv)
            if not conv_only:
                self.add_module(
                    "adn",
                    ADN(
                        ordering=adn_ordering,
                        in_channels=out_channels,
                        act=act,
                        norm=norm,
                        norm_dim=spatial_dims,
                        dropout=dropout,
                        dropout_dim=dropout_dim,
                    ),
                )

    class ConvBlock(nn.Module):
        def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            strides: int = 1,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.conv = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=None,
                    adn_ordering="NDA",
                    act="relu",
                    norm="batch",
                    dropout=dropout,
                ),
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=1,
                    padding=None,
                    adn_ordering="NDA",
                    act="relu",
                    norm="batch",
                    dropout=dropout,
                ),
            )

        def forward(self, x):
            return self.conv(x)

    class UpConv(nn.Module):
        def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            strides: int = 2,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.up = Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_size,
                act="relu",
                adn_ordering="NDA",
                norm="batch",
                dropout=dropout,
                is_transposed=True,
            )

        def forward(self, x):
            return self.up(x)

    class AttentionBlock(nn.Module):
        def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout: float = 0.0):
            super().__init__()
            self.W_g = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=f_g,
                    out_channels=f_int,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    dropout=dropout,
                    conv_only=True,
                ),
                nn.BatchNorm2d(f_int),
            )
            self.W_x = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=f_l,
                    out_channels=f_int,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    dropout=dropout,
                    conv_only=True,
                ),
                nn.BatchNorm2d(f_int),
            )
            self.psi = nn.Sequential(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=f_int,
                    out_channels=1,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    dropout=dropout,
                    conv_only=True,
                ),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=False)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class AttentionLayer(nn.Module):
        def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, submodule: nn.Module, dropout: float = 0.0):
            super().__init__()
            self.attention = AttentionBlock(
                spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
            )
            self.upconv = UpConv(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=in_channels, strides=2)
            self.merge = Convolution(
                spatial_dims=spatial_dims,
                in_channels=2 * in_channels,
                out_channels=in_channels,
                dropout=dropout,
            )
            self.submodule = submodule

        def forward(self, x):
            fromlower = self.upconv(self.submodule(x))
            att = self.attention(g=fromlower, x=x)
            return self.merge(torch_module.cat((att, fromlower), dim=1))

    class AttentionUnet(nn.Module):
        def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.dimensions = spatial_dims
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.channels = channels
            self.strides = strides
            self.kernel_size = kernel_size
            self.dropout = dropout
            self.up_kernel_size = up_kernel_size

            head = ConvBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=channels[0], dropout=dropout)
            reduce_channels = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[0],
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            )

            def _create_block(block_channels: Sequence[int], block_strides: Sequence[int]) -> nn.Module:
                if len(block_channels) > 2:
                    subblock = _create_block(block_channels[1:], block_strides[1:])
                    return AttentionLayer(
                        spatial_dims=spatial_dims,
                        in_channels=block_channels[0],
                        out_channels=block_channels[1],
                        submodule=nn.Sequential(
                            ConvBlock(
                                spatial_dims=spatial_dims,
                                in_channels=block_channels[0],
                                out_channels=block_channels[1],
                                strides=block_strides[0],
                                dropout=self.dropout,
                            ),
                            subblock,
                        ),
                        dropout=dropout,
                    )
                return self._get_bottom_layer(block_channels[0], block_channels[1], block_strides[0])

            encdec = _create_block(self.channels, self.strides)
            self.model = nn.Sequential(head, encdec, reduce_channels)

        def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
            return AttentionLayer(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                submodule=ConvBlock(
                    spatial_dims=self.dimensions,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dropout=self.dropout,
                ),
                dropout=self.dropout,
            )

        def forward(self, x):
            return self.model(x)

    return AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        dropout=dropout,
    )


class DefSynUSSequenceInference(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DeSynUS Sequence Inference"
        parent.categories = ["Ultrasound"]
        parent.dependencies = ["Sequences"]
        parent.contributors = ["OpenAI Codex (generated from DeSynUS inference script)"]
        parent.helpText = (
            "Run DeSynUS CUT + segmentation inference on a 2D ultrasound sequence. "
            "The pipeline matches inference_classification_metric.py "
            "(real US -> CUT netG -> segmentation model)."
        )
        parent.acknowledgementText = "Research code adaptation for 3D Slicer."


class DefSynUSSequenceInferenceWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = DefSynUSSequenceInferenceLogic()

        formLayout = qt.QFormLayout()
        self.layout.addLayout(formLayout)

        self.configPathEdit = ctk.ctkPathLineEdit()
        self.configPathEdit.filters = ctk.ctkPathLineEdit.Files
        self.configPathEdit.nameFilters = ["YAML (*.yml *.yaml)", "All files (*)"]
        self.configPathEdit.toolTip = "DeSynUS YAML config used for training/inference (for n_classes and architecture flags)."
        formLayout.addRow("Config YAML", self.configPathEdit)

        self.segCkptPathEdit = ctk.ctkPathLineEdit()
        self.segCkptPathEdit.filters = ctk.ctkPathLineEdit.Files
        self.segCkptPathEdit.nameFilters = ["PyTorch checkpoint (*.pt *.pth)", "All files (*)"]
        self.segCkptPathEdit.toolTip = "Segmentation checkpoint (.pt) saved from the DeSynUS trainer."
        formLayout.addRow("Seg checkpoint", self.segCkptPathEdit)

        self.cutCkptPathEdit = ctk.ctkPathLineEdit()
        self.cutCkptPathEdit.filters = ctk.ctkPathLineEdit.Files
        self.cutCkptPathEdit.nameFilters = ["PyTorch checkpoint (*.pt *.pth)", "All files (*)"]
        self.cutCkptPathEdit.toolTip = "CUT generator checkpoint (.pt) saved by DeSynUS."
        formLayout.addRow("CUT checkpoint", self.cutCkptPathEdit)

        self.inputSequenceSelector = slicer.qMRMLNodeComboBox()
        self.inputSequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
        self.inputSequenceSelector.noneEnabled = False
        self.inputSequenceSelector.addEnabled = False
        self.inputSequenceSelector.removeEnabled = False
        self.inputSequenceSelector.renameEnabled = True
        self.inputSequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSequenceSelector.toolTip = "Input ultrasound sequence (sequence of scalar/vector volume frames)."
        formLayout.addRow("Input sequence", self.inputSequenceSelector)

        self.outputLabelSequenceSelector = slicer.qMRMLNodeComboBox()
        self.outputLabelSequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
        self.outputLabelSequenceSelector.noneEnabled = False
        self.outputLabelSequenceSelector.addEnabled = True
        self.outputLabelSequenceSelector.removeEnabled = False
        self.outputLabelSequenceSelector.renameEnabled = True
        self.outputLabelSequenceSelector.baseName = "DeSynUSLabelmapSequence"
        self.outputLabelSequenceSelector.selectNodeUponCreation = True
        self.outputLabelSequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.outputLabelSequenceSelector.toolTip = "Output sequence of labelmap volume frames."
        formLayout.addRow("Output label sequence", self.outputLabelSequenceSelector)

        self.outputReconSequenceSelector = slicer.qMRMLNodeComboBox()
        self.outputReconSequenceSelector.nodeTypes = ["vtkMRMLSequenceNode"]
        self.outputReconSequenceSelector.noneEnabled = True
        self.outputReconSequenceSelector.addEnabled = True
        self.outputReconSequenceSelector.removeEnabled = False
        self.outputReconSequenceSelector.renameEnabled = True
        self.outputReconSequenceSelector.baseName = "DeSynUSReconstructedUSSequence"
        self.outputReconSequenceSelector.selectNodeUponCreation = True
        self.outputReconSequenceSelector.setMRMLScene(slicer.mrmlScene)
        self.outputReconSequenceSelector.toolTip = "Optional output sequence of reconstructed US images after CUT."
        formLayout.addRow("Output recon sequence", self.outputReconSequenceSelector)

        self.deviceComboBox = qt.QComboBox()
        self.deviceComboBox.addItems(["auto", "cuda", "cpu"])
        self.deviceComboBox.setCurrentText("auto")
        formLayout.addRow("Device", self.deviceComboBox)

        self.binaryThresholdSpinBox = qt.QDoubleSpinBox()
        self.binaryThresholdSpinBox.minimum = 0.0
        self.binaryThresholdSpinBox.maximum = 1.0
        self.binaryThresholdSpinBox.singleStep = 0.05
        self.binaryThresholdSpinBox.value = 0.5
        self.binaryThresholdSpinBox.toolTip = "Used only when n_classes == 1 (binary segmentation)."
        formLayout.addRow("Binary threshold", self.binaryThresholdSpinBox)

        self.fullProcessingCheckBox = qt.QCheckBox()
        self.fullProcessingCheckBox.checked = False
        self.fullProcessingCheckBox.toolTip = (
            "Apply full post-processing similar to inference_classification_metric.py metrics filtering: "
            "remove small connected prediction islands per class after segmentation. "
            "This improves cleanup but can reduce FPS."
        )
        formLayout.addRow("Full post-processing", self.fullProcessingCheckBox)

        self.cutNetGComboBox = qt.QComboBox()
        self.cutNetGComboBox.addItems(["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"])
        self.cutNetGComboBox.setCurrentText("resnet_9blocks")
        formLayout.addRow("CUT netG", self.cutNetGComboBox)

        self.cutNormGComboBox = qt.QComboBox()
        self.cutNormGComboBox.addItems(["instance", "batch", "none"])
        self.cutNormGComboBox.setCurrentText("instance")
        formLayout.addRow("CUT normG", self.cutNormGComboBox)

        self.cutNgfSpinBox = qt.QSpinBox()
        self.cutNgfSpinBox.minimum = 1
        self.cutNgfSpinBox.maximum = 512
        self.cutNgfSpinBox.value = 64
        formLayout.addRow("CUT ngf", self.cutNgfSpinBox)

        self.keepTempNodesCheckBox = qt.QCheckBox()
        self.keepTempNodesCheckBox.checked = False
        self.keepTempNodesCheckBox.toolTip = "If checked, per-frame temporary nodes stay in scene (debug only)."
        formLayout.addRow("Keep temp nodes", self.keepTempNodesCheckBox)

        self.livePreviewCheckBox = qt.QCheckBox()
        self.livePreviewCheckBox.checked = True
        self.livePreviewCheckBox.toolTip = (
            "Update slice viewers while processing by syncing outputs to a Sequence Browser "
            "and overlaying the predicted labelmap on the input sequence."
        )
        formLayout.addRow("Live preview", self.livePreviewCheckBox)

        self.previewLabelOpacitySpinBox = qt.QDoubleSpinBox()
        self.previewLabelOpacitySpinBox.minimum = 0.0
        self.previewLabelOpacitySpinBox.maximum = 1.0
        self.previewLabelOpacitySpinBox.singleStep = 0.05
        self.previewLabelOpacitySpinBox.value = 0.5
        self.previewLabelOpacitySpinBox.toolTip = "Label overlay opacity for live preview."
        formLayout.addRow("Preview opacity", self.previewLabelOpacitySpinBox)

        self.legendGroupBox = ctk.ctkCollapsibleButton()
        self.legendGroupBox.text = "Label Legend"
        self.legendGroupBox.collapsed = False
        legendLayout = qt.QVBoxLayout(self.legendGroupBox)
        self.legendTextEdit = qt.QPlainTextEdit()
        self.legendTextEdit.readOnly = True
        self.legendTextEdit.maximumHeight = 120
        self.legendTextEdit.setPlainText(self.logic.formatLegendText(DEFAULT_VESSEL_LABEL_LEGEND))
        legendLayout.addWidget(self.legendTextEdit)
        self.layout.addWidget(self.legendGroupBox)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        self.applyButton = qt.QPushButton("Run DeSynUS inference on sequence")
        self.applyButton.toolTip = "Run CUT + segmentation on all frames in the input sequence."
        self.applyButton.clicked.connect(self.onApplyButton)
        self.layout.addWidget(self.applyButton)

        self.layout.addStretch(1)

        self._populateConvenienceDefaults()
        self._applyDefaultSequenceSelections()
        self._updateApplyButtonState()

        for widget in [
            self.configPathEdit,
            self.segCkptPathEdit,
            self.cutCkptPathEdit,
            self.inputSequenceSelector,
            self.outputLabelSequenceSelector,
        ]:
            if hasattr(widget, "currentPathChanged"):
                widget.currentPathChanged.connect(self._updateApplyButtonState)
            if hasattr(widget, "currentNodeChanged"):
                widget.currentNodeChanged.connect(self._updateApplyButtonState)
        if hasattr(self.configPathEdit, "currentPathChanged"):
            self.configPathEdit.currentPathChanged.connect(self._updateLegendPreviewFromConfig)
        self._updateLegendPreviewFromConfig()

    def _updateLegendPreviewFromConfig(self, *args):
        configPath = self.configPathEdit.currentPath
        legend = DEFAULT_VESSEL_LABEL_LEGEND
        try:
            legend = self.logic.guessLabelLegend(configPath)
        except Exception:
            LOGGER.exception("Failed to refresh legend preview from config.")
        self.legendTextEdit.setPlainText(self.logic.formatLegendText(legend))

    def _populateConvenienceDefaults(self):
        repoRoot = HARD_CODED_REPO_ROOT
        if not repoRoot:
            return

        defaultConfig = os.path.join(repoRoot, "config", "config_run_us_multilabel_liver.yml")
        if os.path.exists(defaultConfig):
            self.configPathEdit.currentPath = defaultConfig

        defaultSeg = os.path.join(
            repoRoot,
            "checkpoints",
            "best_checkpoint_seg_renderer_valid_loss_33_exp_name25_5e-06_0.0001_0.0001_epoch=24.pt",
        )
        if os.path.exists(defaultSeg):
            self.segCkptPathEdit.currentPath = defaultSeg

        defaultCut = os.path.join(
            repoRoot,
            "checkpoints",
            "best_checkpoint_CUT_val_loss_33_exp_name25_5e-06_0.0001_0.0001_epoch=24.pt",
        )
        if os.path.exists(defaultCut):
            self.cutCkptPathEdit.currentPath = defaultCut

    def _applyDefaultSequenceSelections(self):
        self._selectDefaultInputSequence(preferNameSubstring="image")
        self._ensureDefaultOutputLabelSequence()

    def _iterSequenceNodes(self):
        scene = slicer.mrmlScene
        if scene is None:
            return []
        try:
            collection = scene.GetNodesByClass("vtkMRMLSequenceNode")
        except Exception:
            return []
        nodes = []
        try:
            collection.InitTraversal()
            while True:
                node = collection.GetNextItemAsObject()
                if node is None:
                    break
                nodes.append(node)
        finally:
            try:
                collection.UnRegister(None)
            except Exception:
                pass
        return nodes

    def _sequenceFirstNonNullDataNode(self, sequenceNode):
        if sequenceNode is None:
            return None
        try:
            count = int(sequenceNode.GetNumberOfDataNodes())
        except Exception:
            return None
        for i in range(count):
            try:
                node = sequenceNode.GetNthDataNode(i)
            except Exception:
                node = None
            if node is not None:
                return node
        return None

    def _isVolumeSequence(self, sequenceNode):
        dataNode = self._sequenceFirstNonNullDataNode(sequenceNode)
        return bool(dataNode is not None and hasattr(dataNode, "IsA") and dataNode.IsA("vtkMRMLVolumeNode"))

    def _selectDefaultInputSequence(self, preferNameSubstring="image"):
        current = self.inputSequenceSelector.currentNode()
        if self._isVolumeSequence(current):
            return

        sequences = self._iterSequenceNodes()
        if not sequences:
            return

        preferredLower = str(preferNameSubstring or "").strip().lower()
        candidates = []
        fallbackCandidates = []
        for seq in sequences:
            if not self._isVolumeSequence(seq):
                continue
            nameLower = (seq.GetName() or "").lower()
            if preferredLower and preferredLower in nameLower:
                candidates.append(seq)
            else:
                fallbackCandidates.append(seq)

        chosen = None
        if candidates:
            # Prefer names that start with "image", then any containing "image", then shortest name.
            candidates.sort(key=lambda n: (0 if (n.GetName() or "").lower().startswith(preferredLower) else 1, len(n.GetName() or "")))
            chosen = candidates[0]
        elif fallbackCandidates:
            fallbackCandidates.sort(key=lambda n: len(n.GetName() or ""))
            chosen = fallbackCandidates[0]

        if chosen is not None:
            try:
                self.inputSequenceSelector.setCurrentNode(chosen)
            except Exception:
                LOGGER.exception("Failed to set default input sequence selection.")

    def _ensureDefaultOutputLabelSequence(self):
        currentInput = self.inputSequenceSelector.currentNode()
        currentOutput = self.outputLabelSequenceSelector.currentNode()
        baseName = getattr(self.outputLabelSequenceSelector, "baseName", None) or "DeSynUSLabelmapSequence"

        def _looks_like_default_output(node):
            if node is None:
                return False
            try:
                name = (node.GetName() or "").lower()
            except Exception:
                name = ""
            return name.startswith(str(baseName).lower())

        shouldCreateNew = (
            currentOutput is None
            or currentOutput == currentInput
            or (not _looks_like_default_output(currentOutput))
        )
        if not shouldCreateNew:
            return

        try:
            newName = slicer.mrmlScene.GenerateUniqueName(str(baseName))
            newNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", newName)
            self.outputLabelSequenceSelector.setCurrentNode(newNode)
        except Exception:
            LOGGER.exception("Failed to create/select default output label sequence.")

    def _updateApplyButtonState(self, *args):
        ready = bool(self.inputSequenceSelector.currentNode() and self.outputLabelSequenceSelector.currentNode())
        self.applyButton.enabled = ready

    def onApplyButton(self):
        params = {
            "configPath": self.configPathEdit.currentPath,
            "segCheckpointPath": self.segCkptPathEdit.currentPath,
            "cutCheckpointPath": self.cutCkptPathEdit.currentPath,
            "inputSequenceNode": self.inputSequenceSelector.currentNode(),
            "outputLabelSequenceNode": self.outputLabelSequenceSelector.currentNode(),
            "outputReconstructedSequenceNode": self.outputReconSequenceSelector.currentNode(),
            "devicePreference": self.deviceComboBox.currentText,
            "binaryThreshold": float(self.binaryThresholdSpinBox.value),
            "applyFullPostProcessing": bool(self.fullProcessingCheckBox.checked),
            "cutNetG": self.cutNetGComboBox.currentText,
            "cutNormG": self.cutNormGComboBox.currentText,
            "cutNgf": int(self.cutNgfSpinBox.value),
            "keepTempNodes": bool(self.keepTempNodesCheckBox.checked),
            "livePreview": bool(self.livePreviewCheckBox.checked),
            "previewLabelOpacity": float(self.previewLabelOpacitySpinBox.value),
        }

        try:
            self.applyButton.enabled = False
            self.statusLabel.text = "Running inference..."
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            summary = self.logic.run(**params)
            self.statusLabel.text = summary
            slicer.util.showStatusMessage(summary, 8000)
        except Exception as exc:
            message = f"DeSynUS sequence inference failed: {exc}"
            LOGGER.exception(message)
            self.statusLabel.text = message
            slicer.util.errorDisplay(f"{message}\n\n{traceback.format_exc()}")
        finally:
            qt.QApplication.restoreOverrideCursor()
            self._updateApplyButtonState()


class DefSynUSSequenceInferenceLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._engine = None
        self._engineKey = None
        self._legendColorTableCacheKey = None
        self._legendColorTableNodeID = None

    def run(
        self,
        configPath: str,
        segCheckpointPath: str,
        cutCheckpointPath: str,
        inputSequenceNode,
        outputLabelSequenceNode,
        outputReconstructedSequenceNode=None,
        devicePreference: str = "auto",
        binaryThreshold: float = 0.5,
        applyFullPostProcessing: bool = False,
        cutNetG: str = "resnet_9blocks",
        cutNormG: str = "instance",
        cutNgf: int = 64,
        keepTempNodes: bool = False,
        livePreview: bool = True,
        previewLabelOpacity: float = 0.5,
    ) -> str:
        if inputSequenceNode is None:
            raise ValueError("Input sequence node is required.")
        if outputLabelSequenceNode is None:
            raise ValueError("Output label sequence node is required.")
        if inputSequenceNode == outputLabelSequenceNode:
            raise ValueError(
                "Input sequence and output label sequence must be different nodes. "
                "Create/select a separate output sequence."
            )
        if outputReconstructedSequenceNode is not None and inputSequenceNode == outputReconstructedSequenceNode:
            raise ValueError(
                "Input sequence and output reconstructed sequence must be different nodes."
            )
        if (
            outputReconstructedSequenceNode is not None
            and outputLabelSequenceNode == outputReconstructedSequenceNode
        ):
            raise ValueError(
                "Output label sequence and output reconstructed sequence must be different nodes."
            )
        repoRoot = HARD_CODED_REPO_ROOT
        if not repoRoot or not os.path.isdir(repoRoot):
            raise ValueError(f"Invalid DeSynUS repo root: {repoRoot}")
        if not segCheckpointPath or not os.path.isfile(segCheckpointPath):
            raise ValueError(f"Segmentation checkpoint not found: {segCheckpointPath}")
        if not cutCheckpointPath or not os.path.isfile(cutCheckpointPath):
            raise ValueError(f"CUT checkpoint not found: {cutCheckpointPath}")

        engine = self._getOrCreateEngine(
            repoRoot=repoRoot,
            configPath=configPath,
            segCheckpointPath=segCheckpointPath,
            cutCheckpointPath=cutCheckpointPath,
            devicePreference=devicePreference,
            cutNetG=cutNetG,
            cutNormG=cutNormG,
            cutNgf=cutNgf,
        )
        labelLegend = self.guessLabelLegend(configPath=configPath, nClasses=getattr(engine, "nClasses", None))
        legendColorTableNode = self._getOrCreateLegendColorTableNode(labelLegend, baseName="DeSynUSLabels")

        frameCount = inputSequenceNode.GetNumberOfDataNodes()
        if frameCount <= 0:
            raise ValueError("Input sequence has no frames.")

        firstFrameIndex, firstFrameNode = self._findFirstNonNullSequenceDataNode(inputSequenceNode)
        if firstFrameNode is None:
            raise ValueError("Input sequence contains no valid data nodes.")
        if not firstFrameNode.IsA("vtkMRMLVolumeNode"):
            className = firstFrameNode.GetClassName() if hasattr(firstFrameNode, "GetClassName") else type(firstFrameNode).__name__
            raise ValueError(
                "Selected input sequence does not contain image frames. "
                f"First frame node type is '{className}'. "
                "Select the ultrasound image sequence (scalar/vector volume), not a transform/tracking sequence."
            )

        self._prepareOutputSequence(outputLabelSequenceNode, inputSequenceNode)
        if outputReconstructedSequenceNode:
            self._prepareOutputSequence(outputReconstructedSequenceNode, inputSequenceNode)

        # Defensive check in case the input sequence was altered externally.
        if inputSequenceNode.GetNumberOfDataNodes() <= 0:
            raise RuntimeError(
                "Input sequence has no frames at runtime. "
                "Verify you selected a populated input sequence and distinct output sequence nodes."
            )

        livePreviewState = None
        if livePreview:
            livePreviewState = self._initializeLivePreview(
                inputSequenceNode=inputSequenceNode,
                outputLabelSequenceNode=outputLabelSequenceNode,
                outputReconstructedSequenceNode=outputReconstructedSequenceNode,
                firstFrameNode=firstFrameNode,
                labelOpacity=previewLabelOpacity,
            )

        progress = slicer.util.createProgressDialog(
            autoClose=False,
            value=0,
            minimum=0,
            maximum=frameCount,
            labelText="DeSynUS inference in progress...",
        )
        loopStartTimeSec = time.perf_counter()
        avgFps = 0.0
        lastUiUpdateSec = 0.0
        uiUpdateIntervalSec = 0.20 if livePreview else 0.40
        minFramesBetweenUiUpdates = 2 if livePreview else 4
        lastUiFrameIndex = -10**9

        reusableLabelFrameNode = None
        reusableReconFrameNode = None
        if not keepTempNodes:
            reusableLabelFrameNode = self._createVolumeNodeFrom2DArray(
                sourceVolumeNode=firstFrameNode,
                array2d=np.zeros((1, 1), dtype=np.uint16),
                nodeClassName="vtkMRMLLabelMapVolumeNode",
                nodeName=f"{outputLabelSequenceNode.GetName()}_tmp_reusable",
            )
            self._applyLegendToLabelVolumeNode(reusableLabelFrameNode, legendColorTableNode)
            if outputReconstructedSequenceNode is not None:
                reusableReconFrameNode = self._createVolumeNodeFrom2DArray(
                    sourceVolumeNode=firstFrameNode,
                    array2d=np.zeros((1, 1), dtype=np.uint8),
                    nodeClassName="vtkMRMLScalarVolumeNode",
                    nodeName=f"{outputReconstructedSequenceNode.GetName()}_tmp_reusable",
                )

        try:
            for i in range(frameCount):
                frameStartTimeSec = time.perf_counter()
                wasCanceledAttr = getattr(progress, "wasCanceled", False)
                isCanceled = wasCanceledAttr() if callable(wasCanceledAttr) else bool(wasCanceledAttr)
                if isCanceled:
                    raise RuntimeError("Inference canceled by user.")

                frameNode = inputSequenceNode.GetNthDataNode(i)
                if frameNode is None:
                    raise RuntimeError(f"Frame {i} is empty.")
                if not frameNode.IsA("vtkMRMLVolumeNode"):
                    className = frameNode.GetClassName() if hasattr(frameNode, "GetClassName") else type(frameNode).__name__
                    raise RuntimeError(
                        f"Frame {i} is not a volume node (got {className}). "
                        "Select an ultrasound image sequence."
                    )

                indexValue = inputSequenceNode.GetNthIndexValue(i)

                label2d, recon2d, meta = engine.inferVolumeNode(
                    frameNode,
                    binaryThreshold=binaryThreshold,
                    returnReconstruction=(outputReconstructedSequenceNode is not None),
                    applyFullPostProcessing=applyFullPostProcessing,
                )

                if reusableLabelFrameNode is not None:
                    labelFrameNode = reusableLabelFrameNode
                    self._updateVolumeNodeFrom2DArray(
                        targetNode=labelFrameNode,
                        sourceVolumeNode=frameNode,
                        array2d=label2d.astype(np.uint16, copy=False),
                    )
                else:
                    labelFrameNode = self._createVolumeNodeFrom2DArray(
                        sourceVolumeNode=frameNode,
                        array2d=label2d.astype(np.uint16, copy=False),
                        nodeClassName="vtkMRMLLabelMapVolumeNode",
                        nodeName=f"{outputLabelSequenceNode.GetName()}_tmp_{i:04d}",
                    )
                    self._applyLegendToLabelVolumeNode(labelFrameNode, legendColorTableNode)
                outputLabelSequenceNode.SetDataNodeAtValue(labelFrameNode, indexValue)
                if reusableLabelFrameNode is None and not keepTempNodes:
                    slicer.mrmlScene.RemoveNode(labelFrameNode)

                if outputReconstructedSequenceNode is not None:
                    if reusableReconFrameNode is not None:
                        reconFrameNode = reusableReconFrameNode
                        self._updateVolumeNodeFrom2DArray(
                            targetNode=reconFrameNode,
                            sourceVolumeNode=frameNode,
                            array2d=recon2d.astype(np.uint8, copy=False),
                        )
                    else:
                        reconFrameNode = self._createVolumeNodeFrom2DArray(
                            sourceVolumeNode=frameNode,
                            array2d=recon2d.astype(np.uint8, copy=False),
                            nodeClassName="vtkMRMLScalarVolumeNode",
                            nodeName=f"{outputReconstructedSequenceNode.GetName()}_tmp_{i:04d}",
                        )
                    outputReconstructedSequenceNode.SetDataNodeAtValue(reconFrameNode, indexValue)
                    if reusableReconFrameNode is None and not keepTempNodes:
                        slicer.mrmlScene.RemoveNode(reconFrameNode)

                frameElapsedSec = max(time.perf_counter() - frameStartTimeSec, 1e-9)
                instFps = 1.0 / frameElapsedSec
                nowSec = time.perf_counter()
                totalElapsedSec = max(nowSec - loopStartTimeSec, 1e-9)
                avgFps = float(i + 1) / totalElapsedSec

                shouldRefreshUi = (
                    i == 0
                    or i == frameCount - 1
                    or (
                        (i - lastUiFrameIndex) >= minFramesBetweenUiUpdates
                        and (nowSec - lastUiUpdateSec) >= uiUpdateIntervalSec
                    )
                )
                if livePreviewState is not None:
                    self._updateLivePreview(
                        state=livePreviewState,
                        inputSequenceNode=inputSequenceNode,
                        outputLabelSequenceNode=outputLabelSequenceNode,
                        outputReconstructedSequenceNode=outputReconstructedSequenceNode,
                        itemNumber=i,
                        labelOpacity=previewLabelOpacity,
                        legendColorTableNode=legendColorTableNode,
                        forceProxySync=shouldRefreshUi,
                    )
                    # Keep the slice viewers advancing continuously through all frames.
                    slicer.app.processEvents()

                if shouldRefreshUi:
                    progress.setValue(i + 1)
                    progress.setLabelText(
                        f"DeSynUS inference: frame {i + 1}/{frameCount} "
                        f"(input {meta['input_size'][0]}x{meta['input_size'][1]}) | "
                        f"FPS {instFps:.1f} (avg {avgFps:.1f})"
                    )
                    slicer.util.showStatusMessage(
                        f"DeSynUS frame {i + 1}/{frameCount} | FPS {instFps:.1f} (avg {avgFps:.1f})",
                        2000,
                    )
                    lastUiUpdateSec = nowSec
                    lastUiFrameIndex = i
        finally:
            progress.close()
            if reusableLabelFrameNode is not None:
                slicer.mrmlScene.RemoveNode(reusableLabelFrameNode)
            if reusableReconFrameNode is not None:
                slicer.mrmlScene.RemoveNode(reusableReconFrameNode)

        return (
            f"Processed {frameCount} frames on {engine.deviceLabel} "
            f"(avg {avgFps:.1f} FPS). "
            f"Label sequence: {outputLabelSequenceNode.GetName()}"
        )

    def _getOrCreateEngine(
        self,
        repoRoot: str,
        configPath: str,
        segCheckpointPath: str,
        cutCheckpointPath: str,
        devicePreference: str,
        cutNetG: str,
        cutNormG: str,
        cutNgf: int,
    ):
        engineKey = (
            os.path.abspath(repoRoot),
            os.path.abspath(configPath) if configPath else "",
            os.path.abspath(segCheckpointPath),
            os.path.abspath(cutCheckpointPath),
            devicePreference,
            cutNetG,
            cutNormG,
            int(cutNgf),
        )
        if self._engine is not None and self._engineKey == engineKey:
            return self._engine

        self._engine = LotusTorchInferenceEngine(
            repoRoot=repoRoot,
            configPath=configPath,
            segCheckpointPath=segCheckpointPath,
            cutCheckpointPath=cutCheckpointPath,
            devicePreference=devicePreference,
            cutNetG=cutNetG,
            cutNormG=cutNormG,
            cutNgf=cutNgf,
        )
        self._engineKey = engineKey
        return self._engine

    def _prepareOutputSequence(self, outputSequenceNode, inputSequenceNode):
        outputSequenceNode.RemoveAllDataNodes()
        outputSequenceNode.SetIndexName(inputSequenceNode.GetIndexName())
        outputSequenceNode.SetIndexUnit(inputSequenceNode.GetIndexUnit())
        outputSequenceNode.SetIndexType(inputSequenceNode.GetIndexType())

    def _sequencesLogic(self):
        sequencesModule = getattr(slicer.modules, "sequences", None)
        if sequencesModule is None:
            return None
        logicMethod = getattr(sequencesModule, "logic", None)
        return logicMethod() if callable(logicMethod) else None

    def _parseConfigNClasses(self, configPath: str):
        if not configPath or not os.path.isfile(configPath):
            return None
        try:
            with open(configPath, "r", encoding="utf-8") as f:
                for rawLine in f:
                    line = rawLine.split("#", 1)[0].strip()
                    if not line or ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key.strip() != "n_classes":
                        continue
                    try:
                        return int(value.strip())
                    except Exception:
                        return None
        except Exception:
            return None
        return None

    def guessLabelLegend(self, configPath: str = "", nClasses: Optional[int] = None):
        nClasses = int(nClasses) if nClasses is not None else self._parseConfigNClasses(configPath)
        if nClasses == 5:
            return dict(DEFAULT_VESSEL_LABEL_LEGEND)
        if nClasses == 1:
            return {
                0: ("BG", (0.0, 0.0, 0.0, 0.0)),
                1: ("Target", (1.0, 0.0, 0.0, 1.0)),
            }
        if nClasses and nClasses > 1:
            legend = {0: ("BG", (0.0, 0.0, 0.0, 0.0))}
            palette = [
                (0.89, 0.10, 0.11, 1.0),
                (0.22, 0.49, 0.72, 1.0),
                (0.30, 0.69, 0.29, 1.0),
                (1.00, 0.50, 0.00, 1.0),
                (0.60, 0.31, 0.64, 1.0),
                (0.65, 0.34, 0.16, 1.0),
                (0.97, 0.51, 0.75, 1.0),
                (0.60, 0.60, 0.60, 1.0),
            ]
            for i in range(1, nClasses):
                legend[i] = (f"Label {i}", palette[(i - 1) % len(palette)])
            return legend
        return dict(DEFAULT_VESSEL_LABEL_LEGEND)

    def formatLegendText(self, legend: Dict[int, Tuple[str, Tuple[float, float, float, float]]]) -> str:
        lines = []
        for labelValue in sorted(legend.keys()):
            name, rgba = legend[labelValue]
            r, g, b, _a = rgba
            hexColor = "#{:02X}{:02X}{:02X}".format(
                int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
            )
            lines.append(f"{labelValue}: {name} ({hexColor})")
        return "\n".join(lines)

    def _legendCacheKey(self, legend: Dict[int, Tuple[str, Tuple[float, float, float, float]]]):
        return tuple(
            (int(k), str(v[0]), tuple(round(float(c), 6) for c in v[1]))
            for k, v in sorted(legend.items(), key=lambda item: item[0])
        )

    def _getOrCreateLegendColorTableNode(
        self,
        legend: Dict[int, Tuple[str, Tuple[float, float, float, float]]],
        baseName: str = "DeSynUSLabels",
    ):
        cacheKey = self._legendCacheKey(legend)
        if self._legendColorTableNodeID and self._legendColorTableCacheKey == cacheKey:
            node = slicer.mrmlScene.GetNodeByID(self._legendColorTableNodeID)
            if node is not None:
                return node

        colorNode = slicer.util.getFirstNodeByClassByName("vtkMRMLColorTableNode", baseName)
        if colorNode is None:
            colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", baseName)

        colorNode.SetTypeToUser()
        maxLabel = max(int(k) for k in legend.keys()) if legend else 0
        colorNode.SetNumberOfColors(maxLabel + 1)
        try:
            colorNode.NamesInitialisedOn()
        except Exception:
            pass

        # Initialize all entries transparent/unknown first.
        for i in range(maxLabel + 1):
            try:
                colorNode.SetColor(i, f"Label {i}", 0.0, 0.0, 0.0, 0.0)
            except TypeError:
                colorNode.SetColor(i, 0.0, 0.0, 0.0, 0.0)

        for labelValue, (labelName, rgba) in legend.items():
            r, g, b, a = [float(c) for c in rgba]
            try:
                colorNode.SetColor(int(labelValue), str(labelName), r, g, b, a)
            except TypeError:
                colorNode.SetColor(int(labelValue), r, g, b, a)
                try:
                    colorNode.SetColorName(int(labelValue), str(labelName))
                except Exception:
                    pass

        self._legendColorTableCacheKey = cacheKey
        self._legendColorTableNodeID = colorNode.GetID()
        return colorNode

    def _applyLegendToLabelVolumeNode(self, labelVolumeNode, colorTableNode):
        if labelVolumeNode is None or colorTableNode is None:
            return
        try:
            labelVolumeNode.CreateDefaultDisplayNodes()
            displayNode = labelVolumeNode.GetDisplayNode()
            if displayNode is None:
                return
            displayNode.SetAndObserveColorNodeID(colorTableNode.GetID())
            try:
                displayNode.SetInterpolate(False)
            except Exception:
                pass
            self._ensureColorLegendVisible(labelVolumeNode)
        except Exception:
            LOGGER.exception("Failed to apply legend color table to label volume node.")

    def _ensureColorLegendVisible(self, displayableNode):
        try:
            colorsModule = getattr(slicer.modules, "colors", None)
            if colorsModule is None:
                return
            colorsLogicFn = getattr(colorsModule, "logic", None)
            colorsLogic = colorsLogicFn() if callable(colorsLogicFn) else None
            if colorsLogic is None:
                return

            colorLegendNode = None
            # Try a few possible helper names across Slicer versions.
            for methodName in [
                "AddDefaultColorLegendDisplayNode",
                "CreateDefaultColorLegendDisplayNode",
                "CreateColorLegendDisplayNode",
            ]:
                method = getattr(colorsLogic, methodName, None)
                if callable(method):
                    try:
                        colorLegendNode = method(displayableNode)
                    except TypeError:
                        try:
                            colorLegendNode = method(displayableNode.GetDisplayNode())
                        except Exception:
                            pass
                    if colorLegendNode is not None:
                        break

            if colorLegendNode is None and hasattr(colorsLogic, "GetColorLegendDisplayNode"):
                colorLegendNode = colorsLogic.GetColorLegendDisplayNode(displayableNode)

            if colorLegendNode is not None:
                try:
                    colorLegendNode.SetVisibility(True)
                except Exception:
                    pass
                try:
                    colorLegendNode.SetTitleText("DeSynUS labels")
                except Exception:
                    pass
                try:
                    colorLegendNode.SetMaxNumberOfColors(16)
                except Exception:
                    pass
        except Exception:
            # Legend display is optional; color table names still work in many Slicer widgets.
            LOGGER.debug("Color legend display node creation not available in this Slicer build.", exc_info=True)

    def _initializeLivePreview(
        self,
        inputSequenceNode,
        outputLabelSequenceNode,
        outputReconstructedSequenceNode,
        firstFrameNode,
        labelOpacity: float,
    ):
        sequencesLogic = self._sequencesLogic()
        if sequencesLogic is None:
            LOGGER.warning("Sequences logic is unavailable; live preview is disabled.")
            return None

        browserNode = sequencesLogic.GetFirstBrowserNodeForSequenceNode(inputSequenceNode)
        createdBrowser = False
        if browserNode is None:
            browserNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSequenceBrowserNode",
                slicer.mrmlScene.GenerateUniqueName(f"{inputSequenceNode.GetName()} DeSynUS browser"),
            )
            browserNode.SetAndObserveMasterSequenceNodeID(inputSequenceNode.GetID())
            createdBrowser = True

        # Keep output sequences synchronized with the input browser so proxy nodes update as frames are added.
        self._ensureSequenceInBrowser(browserNode, outputLabelSequenceNode)
        if outputReconstructedSequenceNode is not None:
            self._ensureSequenceInBrowser(browserNode, outputReconstructedSequenceNode)

        try:
            sequencesLogic.UpdateAllProxyNodes()
        except Exception:
            # Fallback for older/newer wrapper differences
            try:
                sequencesLogic.UpdateProxyNodesFromSequences(browserNode)
            except Exception:
                LOGGER.exception("Failed to initialize Sequence Browser proxy nodes for live preview.")

        self._activateSequenceBrowser(browserNode)

        inputProxyNode = browserNode.GetProxyNode(inputSequenceNode)
        if inputProxyNode is None:
            inputProxyNode = firstFrameNode

        if inputProxyNode is not None and inputProxyNode.IsA("vtkMRMLVolumeNode"):
            try:
                slicer.util.setSliceViewerLayers(background=inputProxyNode)
            except Exception:
                LOGGER.exception("Failed to set input sequence proxy as slice viewer background.")

        return {
            "browserNode": browserNode,
            "sequencesLogic": sequencesLogic,
            "createdBrowser": createdBrowser,
            "inputProxyNode": inputProxyNode,
            "labelOpacity": float(labelOpacity),
            "lastLabelProxyNodeId": None,
            "labelProxyDisplayInitialized": False,
            "sliceViewerLayersInitialized": False,
            "lastInputProxyNodeId": None,
            "lastAppliedLabelOpacity": None,
        }

    def _ensureSequenceInBrowser(self, browserNode, sequenceNode):
        if browserNode is None or sequenceNode is None:
            return
        try:
            proxyNode = browserNode.GetProxyNode(sequenceNode)
            if proxyNode is None:
                browserNode.AddSynchronizedSequenceNode(sequenceNode)
                proxyNode = browserNode.GetProxyNode(sequenceNode)
            try:
                browserNode.SetOverwriteProxyName(sequenceNode, True)
            except Exception:
                # Optional convenience API; safe to ignore if unavailable.
                pass
            if proxyNode is not None and hasattr(proxyNode, "CreateDefaultDisplayNodes"):
                proxyNode.CreateDefaultDisplayNodes()
        except Exception:
            LOGGER.exception("Failed to add sequence '%s' to Sequence Browser live preview.", sequenceNode.GetName())

    def _activateSequenceBrowser(self, browserNode):
        if browserNode is None:
            return
        try:
            if hasattr(slicer.modules, "sequences"):
                try:
                    slicer.modules.sequences.toolBar().setActiveBrowserNode(browserNode)
                except Exception:
                    try:
                        slicer.modules.sequences.setToolBarActiveBrowserNode(browserNode)
                    except Exception:
                        pass
                try:
                    slicer.modules.sequences.showSequenceBrowser(browserNode)
                except Exception:
                    try:
                        slicer.modules.sequences.setToolBarVisible(True)
                    except Exception:
                        pass
        except Exception:
            LOGGER.exception("Failed to activate Sequence Browser for live preview.")

    def _updateLivePreview(
        self,
        state,
        inputSequenceNode,
        outputLabelSequenceNode,
        outputReconstructedSequenceNode,
        itemNumber: int,
        labelOpacity: float,
        legendColorTableNode=None,
        forceProxySync: bool = False,
    ):
        browserNode = state.get("browserNode")
        sequencesLogic = state.get("sequencesLogic")
        if browserNode is None or sequencesLogic is None:
            return

        try:
            browserNode.SetSelectedItemNumber(int(itemNumber))
            if forceProxySync:
                try:
                    sequencesLogic.UpdateProxyNodesFromSequences(browserNode)
                except Exception:
                    sequencesLogic.UpdateAllProxyNodes()

            inputProxyNode = browserNode.GetProxyNode(inputSequenceNode) or state.get("inputProxyNode")
            labelProxyNode = browserNode.GetProxyNode(outputLabelSequenceNode)

            if labelProxyNode is not None:
                currentLabelProxyId = labelProxyNode.GetID() if hasattr(labelProxyNode, "GetID") else str(id(labelProxyNode))
                if state.get("lastLabelProxyNodeId") != currentLabelProxyId:
                    state["lastLabelProxyNodeId"] = currentLabelProxyId
                    state["labelProxyDisplayInitialized"] = False
                    state["sliceViewerLayersInitialized"] = False

                if (
                    not state.get("labelProxyDisplayInitialized", False)
                    and hasattr(labelProxyNode, "CreateDefaultDisplayNodes")
                ):
                    labelProxyNode.CreateDefaultDisplayNodes()
                    if legendColorTableNode is not None:
                        self._applyLegendToLabelVolumeNode(labelProxyNode, legendColorTableNode)
                    state["labelProxyDisplayInitialized"] = True

            currentInputProxyId = None
            if inputProxyNode is not None:
                currentInputProxyId = inputProxyNode.GetID() if hasattr(inputProxyNode, "GetID") else str(id(inputProxyNode))
            if state.get("lastInputProxyNodeId") != currentInputProxyId:
                state["lastInputProxyNodeId"] = currentInputProxyId
                state["sliceViewerLayersInitialized"] = False

            opacityChanged = state.get("lastAppliedLabelOpacity") != float(labelOpacity)
            if opacityChanged:
                state["lastAppliedLabelOpacity"] = float(labelOpacity)
                state["sliceViewerLayersInitialized"] = False

            if (
                not state.get("sliceViewerLayersInitialized", False)
                and inputProxyNode is not None
                and labelProxyNode is not None
            ):
                slicer.util.setSliceViewerLayers(
                    background=inputProxyNode,
                    label=labelProxyNode,
                    labelOpacity=float(labelOpacity),
                )
                state["sliceViewerLayersInitialized"] = True
            elif not state.get("sliceViewerLayersInitialized", False) and inputProxyNode is not None:
                slicer.util.setSliceViewerLayers(background=inputProxyNode)
                state["sliceViewerLayersInitialized"] = True
        except Exception:
            LOGGER.exception("Live preview update failed at frame %s", itemNumber)

    def _findFirstNonNullSequenceDataNode(self, sequenceNode):
        count = sequenceNode.GetNumberOfDataNodes()
        for i in range(count):
            node = sequenceNode.GetNthDataNode(i)
            if node is not None:
                return i, node
        return None, None

    def _createVolumeNodeFrom2DArray(self, sourceVolumeNode, array2d: np.ndarray, nodeClassName: str, nodeName: str):
        if array2d.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {array2d.shape}")
        outNode = slicer.mrmlScene.AddNewNodeByClass(nodeClassName, nodeName)
        outNode.SetHideFromEditors(True)
        self._updateVolumeNodeFrom2DArray(outNode, sourceVolumeNode, array2d)
        return outNode

    def _updateVolumeNodeFrom2DArray(self, targetNode, sourceVolumeNode, array2d: np.ndarray):
        if array2d.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {array2d.shape}")
        outArray = array2d[np.newaxis, ...]  # KJI with one slice
        reusedBuffer = False
        try:
            existingArray = slicer.util.arrayFromVolume(targetNode)
            if existingArray is not None and tuple(existingArray.shape) == tuple(outArray.shape):
                existingArray[...] = outArray
                arrayModifiedFn = getattr(slicer.util, "arrayFromVolumeModified", None)
                if callable(arrayModifiedFn):
                    arrayModifiedFn(targetNode)
                elif hasattr(targetNode, "Modified"):
                    targetNode.Modified()
                reusedBuffer = True
        except Exception:
            reusedBuffer = False

        if not reusedBuffer:
            slicer.util.updateVolumeFromArray(targetNode, outArray)
        self._copyVolumeGeometry(sourceVolumeNode, targetNode)

    def _copyVolumeGeometry(self, sourceNode, targetNode):
        ijkToRAS = vtk.vtkMatrix4x4()
        sourceNode.GetIJKToRASMatrix(ijkToRAS)
        targetNode.SetIJKToRASMatrix(ijkToRAS)
        origin = sourceNode.GetOrigin()
        spacing = sourceNode.GetSpacing()
        targetNode.SetOrigin(origin[0], origin[1], origin[2])
        targetNode.SetSpacing(spacing[0], spacing[1], spacing[2])


class LotusTorchInferenceEngine:
    TARGET_SIZE = (256, 256)
    MONAI_CHANNELS = (32, 64, 128, 256, 512)
    FULL_POSTPROCESSING_MIN_ISLAND_SIZE = 50

    def __init__(
        self,
        repoRoot: str,
        configPath: str,
        segCheckpointPath: str,
        cutCheckpointPath: str,
        devicePreference: str = "auto",
        cutNetG: str = "resnet_9blocks",
        cutNormG: str = "instance",
        cutNgf: int = 64,
    ):
        self.repoRoot = os.path.abspath(repoRoot)
        self.configPath = os.path.abspath(configPath) if configPath else ""
        self.segCheckpointPath = os.path.abspath(segCheckpointPath)
        self.cutCheckpointPath = os.path.abspath(cutCheckpointPath)
        self.cutNetGName = cutNetG
        self.cutNormG = cutNormG
        self.cutNgf = int(cutNgf)

        self._ensureRepoOnPath()
        self.torch = self._importTorch()
        self.Image = self._importPILImage()
        self.device = self._resolveDevice(devicePreference)
        self.deviceLabel = str(self.device)
        try:
            if self.device.type == "cuda":
                self.torch.backends.cudnn.benchmark = True
                if hasattr(self.torch.backends, "cuda") and hasattr(self.torch.backends.cuda, "matmul"):
                    self.torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(self.torch.backends, "cudnn"):
                    self.torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.config = self._loadConfig(self.configPath) if self.configPath else {}
        self.nClasses = int(self._cfg("n_classes", 1))
        self.outerModelMonai = self._asBool(self._cfg("outer_model_monai", True))
        self._scipyNdimage = self._tryImportScipyNdimage()
        self._signalMaskCacheByShape = {}
        self._signalMaskRefreshCounterByShape = {}
        self._signalMaskRefreshEveryNFrames = 12

        self.segModel = self._buildSegModel().to(self.device)
        self.cutNetG = self._buildCutGenerator().to(self.device)

        self._loadSegWeights()
        self._loadCutWeights()

        self.segModel.eval()
        self.cutNetG.eval()

        try:
            if self.device.type == "cuda":
                channelsLast = self.torch.channels_last
                self.segModel = self.segModel.to(memory_format=channelsLast)
                self.cutNetG = self.cutNetG.to(memory_format=channelsLast)
        except Exception:
            LOGGER.debug("Could not switch models to channels_last memory format.", exc_info=True)

    def _ensureRepoOnPath(self):
        if self.repoRoot not in sys.path:
            sys.path.insert(0, self.repoRoot)

    def _importTorch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is not available in this Slicer Python environment. "
                "Install torch (and torchvision, monai if needed) into Slicer's Python."
            ) from exc
        return torch

    def _importPILImage(self):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for image resizing in DeSynUSSequenceInference.") from exc
        return Image

    def _tryImportScipyNdimage(self):
        try:
            from scipy import ndimage  # type: ignore
            return ndimage
        except Exception:
            return None

    def _resolveDevice(self, devicePreference: str):
        pref = (devicePreference or "auto").lower()
        if pref == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        if pref == "cuda":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
            return self.torch.device("cuda")
        if pref == "cpu":
            return self.torch.device("cpu")
        raise ValueError(f"Unsupported device preference: {devicePreference}")

    def _cfg(self, key: str, default):
        return self.config.get(key, default)

    def _asBool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _loadConfig(self, configPath: str) -> Dict[str, object]:
        if not os.path.isfile(configPath):
            raise ValueError(f"Config file not found: {configPath}")

        try:
            import yaml  # type: ignore

            with open(configPath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Config YAML did not parse to a dictionary.")
            return data
        except Exception:
            return self._parseYamlLikeConfigFallback(configPath)

    def _parseYamlLikeConfigFallback(self, configPath: str) -> Dict[str, object]:
        config: Dict[str, object] = {}
        with open(configPath, "r", encoding="utf-8") as f:
            for rawLine in f:
                line = rawLine.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                config[key] = self._parseScalar(value)
        return config

    def _parseScalar(self, value: str):
        if value == "":
            return ""
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        try:
            if "." in value or "e" in lower:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _buildSegModel(self):
        if self.outerModelMonai:
            # Use a local lightweight implementation with MONAI-compatible module names.
            # This avoids importing MONAI in Slicer Python environments where MONAI may fail
            # during package-level auto-import (observed with some Python 3.12 builds).
            model = _build_lite_monai_attention_unet(
                self.torch,
                spatial_dims=2,
                in_channels=1,
                out_channels=self.nClasses,
                channels=self.MONAI_CHANNELS,
                strides=(2, 2, 2, 2),
            )
            return model

        from types import SimpleNamespace
        from models.unet_2d import OriginalUNet

        hparams = SimpleNamespace(n_classes=self.nClasses, dropout=False, dropout_ratio=0.0)
        return OriginalUNet(hparams=hparams)

    def _buildCutGenerator(self):
        from cut.models.networks import define_G

        netG = define_G(
            input_nc=1,
            output_nc=1,
            ngf=self.cutNgf,
            netG=self.cutNetGName,
            norm=self.cutNormG,
            use_dropout=False,  # LOTUSOptions default no_dropout=True -> generator dropout disabled
            init_type="xavier",
            init_gain=0.02,
            no_antialias=False,
            no_antialias_up=False,
            gpu_ids=[],
            opt=None,
        )
        return netG

    def _loadSegWeights(self):
        segState = self.torch.load(self.segCheckpointPath, map_location=self.device)
        if not isinstance(segState, dict):
            raise RuntimeError("Unexpected segmentation checkpoint format (expected state_dict dictionary).")

        candidatePrefixes = ("outer_model.", "module.outer_model.")
        outerState = self._extractPrefixedStateDict(segState, candidatePrefixes)

        if outerState:
            missing, unexpected = self.segModel.load_state_dict(outerState, strict=True)
        else:
            missing, unexpected = self.segModel.load_state_dict(segState, strict=False)

        if unexpected:
            LOGGER.warning("Unexpected segmentation checkpoint keys: %s", list(unexpected)[:20])
        # Missing keys may appear if strict=False and checkpoint contains renderer keys only filtered out.
        if missing:
            LOGGER.info("Missing segmentation keys after load (first 20): %s", list(missing)[:20])

    def _loadCutWeights(self):
        cutState = self.torch.load(self.cutCheckpointPath, map_location=self.device)
        if not isinstance(cutState, dict):
            raise RuntimeError("Unexpected CUT checkpoint format (expected state_dict dictionary).")
        cleaned = {k.replace("module.", "", 1): v for k, v in cutState.items()}
        missing, unexpected = self.cutNetG.load_state_dict(cleaned, strict=True)
        if unexpected:
            LOGGER.warning("Unexpected CUT checkpoint keys: %s", list(unexpected)[:20])
        if missing:
            LOGGER.warning("Missing CUT checkpoint keys: %s", list(missing)[:20])

    def _extractPrefixedStateDict(self, stateDict: Dict[str, object], prefixes: Tuple[str, ...]) -> Dict[str, object]:
        extracted = {}
        for key, value in stateDict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    extracted[key[len(prefix) :]] = value
                    break
        return extracted

    def inferVolumeNode(
        self,
        volumeNode,
        binaryThreshold: float = 0.5,
        returnReconstruction: bool = True,
        applyFullPostProcessing: bool = False,
    ):
        if volumeNode is None:
            raise ValueError("inferVolumeNode received None.")
        if not hasattr(volumeNode, "GetImageData"):
            className = volumeNode.GetClassName() if hasattr(volumeNode, "GetClassName") else type(volumeNode).__name__
            raise TypeError(
                f"inferVolumeNode expected a volume node, got '{className}'. "
                "Select an ultrasound image sequence (volume frames), not a transform sequence."
            )
        frameArray = slicer.util.arrayFromVolume(volumeNode)
        frame2d = self._volumeArrayTo2D(frameArray)
        originalH, originalW = frame2d.shape
        signalMask2d = self._getUsSignalMaskCached(frame2d)

        inputTensor = self._preprocessInput(frame2d)
        if getattr(self.device, "type", "") == "cuda":
            try:
                inputTensor = inputTensor.contiguous(memory_format=self.torch.channels_last)
            except Exception:
                pass

        ampContext = contextlib.nullcontext()
        if getattr(self.device, "type", "") == "cuda":
            try:
                ampContext = self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            except Exception:
                ampContext = contextlib.nullcontext()

        with self.torch.inference_mode():
            with ampContext:
                reconstructed = self.cutNetG(inputTensor)
                reconstructed01 = (reconstructed / 2.0) + 0.5
                reconstructed01 = self.torch.clamp(reconstructed01, 0.0, 1.0)
                segLogits = self.segModel(reconstructed01)

        label256 = self._logitsToLabelMap(segLogits, binaryThreshold=binaryThreshold)
        label2d = self._resizeLabelToOriginalFast(label256, (originalW, originalH))
        if signalMask2d is not None and signalMask2d.shape == label2d.shape:
            label2d = label2d.copy()
            label2d[~signalMask2d] = 0
        if applyFullPostProcessing:
            label2d = self._applyClassificationMetricLikeFiltering(
                label2d,
                minIslandSize=int(self.FULL_POSTPROCESSING_MIN_ISLAND_SIZE),
            )

        recon2d = None
        if returnReconstruction:
            recon2d = self._resizeGrayTensorToOriginalFast(reconstructed01, (originalW, originalH))

        return label2d, recon2d, {"input_size": (originalW, originalH)}

    def _volumeArrayTo2D(self, volumeArray: np.ndarray) -> np.ndarray:
        arr = np.asarray(volumeArray)
        if arr.ndim == 2:
            return self._convertToGrayscale2D(arr)
        if arr.ndim == 3:
            # Scalar 2D frame in Slicer is usually [1, H, W].
            if arr.shape[0] == 1:
                return self._convertToGrayscale2D(arr[0])
            # Some datasets may be stored as [H, W, C].
            if arr.shape[-1] in (3, 4):
                return self._convertToGrayscale2D(arr)
            raise ValueError(
                f"Expected a single-slice 2D frame, got scalar volume shape {arr.shape}. "
                "Use a sequence of 2D frames (1xHxW) for this module."
            )
        if arr.ndim == 4:
            # Vector volume frame often looks like [1, H, W, C].
            if arr.shape[0] == 1 and arr.shape[-1] in (3, 4):
                return self._convertToGrayscale2D(arr[0])
            raise ValueError(f"Unsupported vector frame shape {arr.shape}")
        raise ValueError(f"Unsupported frame array shape {arr.shape}")

    def _convertToGrayscale2D(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        else:
            raise ValueError(f"Cannot convert array of shape {arr.shape} to grayscale.")
        return gray.astype(np.float32, copy=False)

    def _preprocessInput(self, gray2d: np.ndarray):
        img8 = self._toUint8(gray2d)
        try:
            tensor = self.torch.from_numpy(np.ascontiguousarray(img8)).to(self.device, dtype=self.torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0
            tensor = self.torch.nn.functional.interpolate(
                tensor,
                size=(int(self.TARGET_SIZE[1]), int(self.TARGET_SIZE[0])),
                mode="bicubic",
                align_corners=False,
            )
            # Dataset transform in real_us_dataset_with_gt.py: Normalize((0.5,), (0.5,))
            return (tensor - 0.5) / 0.5
        except Exception:
            pil = self.Image.fromarray(img8, mode="L")
            pil = pil.resize(self.TARGET_SIZE, self._pilBicubic())
            arr = np.array(pil, dtype=np.float32) / 255.0
            tensor = self.torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
            return (tensor - 0.5) / 0.5

    def _logitsToLabelMap(self, logits, binaryThreshold: float) -> np.ndarray:
        logits = logits.detach().to("cpu")
        channels = int(logits.shape[1])
        if channels == 1:
            prob = self.torch.sigmoid(logits)
            label = (prob >= float(binaryThreshold)).to(self.torch.uint8)
            return label.squeeze().numpy()
        label = self.torch.argmax(logits, dim=1).to(self.torch.int16)
        return label.squeeze().numpy()

    def _resizeLabelToOriginal(self, label256: np.ndarray, sizeWH: Tuple[int, int]) -> np.ndarray:
        if np.max(label256) <= 255:
            pil = self.Image.fromarray(label256.astype(np.uint8), mode="L")
        else:
            pil = self.Image.fromarray(label256.astype(np.uint16))
        pil = pil.resize(sizeWH, self._pilNearest())
        return np.array(pil)

    def _resizeLabelToOriginalFast(self, label256: np.ndarray, sizeWH: Tuple[int, int]) -> np.ndarray:
        targetW, targetH = int(sizeWH[0]), int(sizeWH[1])
        if label256.shape == (targetH, targetW):
            return np.asarray(label256)
        try:
            tensor = self.torch.from_numpy(np.ascontiguousarray(label256)).unsqueeze(0).unsqueeze(0).to(self.torch.float32)
            resized = self.torch.nn.functional.interpolate(tensor, size=(targetH, targetW), mode="nearest")
            out = resized.squeeze().to(self.torch.int16).numpy()
            return out
        except Exception:
            return self._resizeLabelToOriginal(label256, (targetW, targetH))

    def _resizeGrayToOriginal(self, gray01: np.ndarray, sizeWH: Tuple[int, int]) -> np.ndarray:
        gray01 = np.clip(gray01, 0.0, 1.0)
        arr8 = (gray01 * 255.0).round().astype(np.uint8)
        pil = self.Image.fromarray(arr8, mode="L")
        pil = pil.resize(sizeWH, self._pilBicubic())
        return np.array(pil, dtype=np.uint8)

    def _resizeGrayTensorToOriginalFast(self, grayTensor01, sizeWH: Tuple[int, int]) -> np.ndarray:
        targetW, targetH = int(sizeWH[0]), int(sizeWH[1])
        try:
            tensor = grayTensor01
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-2:] != (targetH, targetW):
                tensor = self.torch.nn.functional.interpolate(
                    tensor,
                    size=(targetH, targetW),
                    mode="bicubic",
                    align_corners=False,
                )
            tensor = self.torch.clamp(tensor, 0.0, 1.0)
            arr8 = (tensor.squeeze().detach().to("cpu") * 255.0).round().to(self.torch.uint8).numpy()
            return np.asarray(arr8, dtype=np.uint8)
        except Exception:
            gray01 = grayTensor01.detach().cpu().squeeze().numpy()
            return self._resizeGrayToOriginal(gray01, (targetW, targetH))

    def _toUint8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            return arr

        finiteMask = np.isfinite(arr)
        if not np.any(finiteMask):
            return np.zeros(arr.shape, dtype=np.uint8)

        valid = arr[finiteMask].astype(np.float32)
        minVal = float(valid.min())
        maxVal = float(valid.max())
        if maxVal <= minVal:
            out = np.zeros(arr.shape, dtype=np.uint8)
            out[finiteMask] = 0
            return out

        scaled = np.zeros(arr.shape, dtype=np.float32)
        scaled[finiteMask] = (arr[finiteMask].astype(np.float32) - minVal) / (maxVal - minVal)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)

    def _applyClassificationMetricLikeFiltering(self, labelMap2d: np.ndarray, minIslandSize: int = 50) -> np.ndarray:
        """
        Mimic the prediction-side island filtering in utils/classification_metrics.py:
        for each foreground class, remove connected components with area < size_threshold.
        """
        minIslandSize = int(minIslandSize)
        if minIslandSize <= 1:
            return np.asarray(labelMap2d)

        labels = np.asarray(labelMap2d)
        filtered = labels.copy()
        uniqueLabels = [int(v) for v in np.unique(labels) if int(v) > 0]
        if not uniqueLabels:
            return filtered

        for classId in uniqueLabels:
            classMask = labels == classId
            cleanedClassMask = self._removeSmallConnectedComponents2D(classMask, minIslandSize=minIslandSize)
            if cleanedClassMask is classMask:
                continue
            filtered[classMask & (~cleanedClassMask)] = 0
        return filtered

    def _removeSmallConnectedComponents2D(self, mask: np.ndarray, minIslandSize: int) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return mask
        minIslandSize = max(int(minIslandSize), 1)

        if self._scipyNdimage is not None:
            try:
                structure = np.ones((3, 3), dtype=np.uint8)
                labeled, numComponents = self._scipyNdimage.label(mask, structure=structure)
                if numComponents <= 0:
                    return np.zeros_like(mask, dtype=bool)
                counts = np.bincount(labeled.ravel())
                if counts.size <= 1:
                    return np.zeros_like(mask, dtype=bool)
                keep = counts >= minIslandSize
                keep[0] = False
                return keep[labeled]
            except Exception:
                LOGGER.debug(
                    "scipy.ndimage component filtering failed; falling back to Python implementation.",
                    exc_info=True,
                )

        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        out = mask.copy()
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            stack = [(y0, x0)]
            coords = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                yMin = 0 if y == 0 else y - 1
                yMax = h if y + 2 > h else y + 2
                xMin = 0 if x == 0 else x - 1
                xMax = w if x + 2 > w else x + 2
                for ny in range(yMin, yMax):
                    for nx in range(xMin, xMax):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(coords) < minIslandSize:
                ysComponent = [p[0] for p in coords]
                xsComponent = [p[1] for p in coords]
                out[ysComponent, xsComponent] = False
        return out

    def _getUsSignalMaskCached(self, gray2d: np.ndarray) -> np.ndarray:
        shapeKey = tuple(int(v) for v in gray2d.shape)
        refreshCounter = int(self._signalMaskRefreshCounterByShape.get(shapeKey, 0))
        cachedMask = self._signalMaskCacheByShape.get(shapeKey)
        shouldRefresh = cachedMask is None or refreshCounter <= 0

        if shouldRefresh:
            cachedMask = self._computeUsSignalMask(gray2d)
            self._signalMaskCacheByShape[shapeKey] = cachedMask
            refreshCounter = int(self._signalMaskRefreshEveryNFrames)

        self._signalMaskRefreshCounterByShape[shapeKey] = max(refreshCounter - 1, 0)
        return cachedMask

    def _computeUsSignalMask(self, gray2d: np.ndarray) -> np.ndarray:
        """
        Build a binary mask of the ultrasound signal region and suppress predictions outside it.
        Strategy: threshold non-black pixels, keep largest connected component, fill internal holes.
        """
        gray8 = self._toUint8(gray2d)

        # Prefer a small threshold to reject compression noise and UI overlays on black background.
        mask = gray8 > 3
        if int(mask.sum()) < max(32, int(mask.size * 0.02)):
            mask = gray8 > 0

        if not np.any(mask):
            return np.ones_like(gray8, dtype=bool)

        if self._scipyNdimage is not None:
            try:
                structure = np.ones((3, 3), dtype=np.uint8)
                labeled, numComponents = self._scipyNdimage.label(mask, structure=structure)
                if numComponents > 0:
                    counts = np.bincount(labeled.ravel())
                    if counts.size > 1:
                        counts[0] = 0
                        largestLabel = int(np.argmax(counts))
                        if largestLabel > 0:
                            mask = labeled == largestLabel
                    mask = self._scipyNdimage.binary_fill_holes(mask).astype(bool)
                    return mask
            except Exception:
                LOGGER.debug(
                    "scipy.ndimage-based US mask extraction failed; falling back to Python implementation.",
                    exc_info=True,
                )

        mask = self._largestConnectedComponent2D(mask)
        if not np.any(mask):
            return np.ones_like(gray8, dtype=bool)

        mask = self._fillHoles2D(mask)
        return mask

    def _largestConnectedComponent2D(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        best_coords = None
        best_size = 0

        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0]:
                continue
            visited[y0, x0] = True
            stack = [(y0, x0)]
            coords = []
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                y_min = 0 if y == 0 else y - 1
                y_max = h if y + 2 > h else y + 2
                x_min = 0 if x == 0 else x - 1
                x_max = w if x + 2 > w else x + 2
                for ny in range(y_min, y_max):
                    for nx in range(x_min, x_max):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(coords) > best_size:
                best_size = len(coords)
                best_coords = coords

        out = np.zeros_like(mask, dtype=bool)
        if best_coords:
            ys_best = [p[0] for p in best_coords]
            xs_best = [p[1] for p in best_coords]
            out[ys_best, xs_best] = True
        return out

    def _fillHoles2D(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in a binary mask using flood fill from the image border on the inverted mask.
        """
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        inv = ~mask
        visited = np.zeros((h, w), dtype=bool)
        stack = []

        # Seed flood-fill with border pixels in the inverted mask (outside background).
        for x in range(w):
            if inv[0, x] and not visited[0, x]:
                visited[0, x] = True
                stack.append((0, x))
            if inv[h - 1, x] and not visited[h - 1, x]:
                visited[h - 1, x] = True
                stack.append((h - 1, x))
        for y in range(h):
            if inv[y, 0] and not visited[y, 0]:
                visited[y, 0] = True
                stack.append((y, 0))
            if inv[y, w - 1] and not visited[y, w - 1]:
                visited[y, w - 1] = True
                stack.append((y, w - 1))

        while stack:
            y, x = stack.pop()
            y_min = 0 if y == 0 else y - 1
            y_max = h if y + 2 > h else y + 2
            x_min = 0 if x == 0 else x - 1
            x_max = w if x + 2 > w else x + 2
            for ny in range(y_min, y_max):
                for nx in range(x_min, x_max):
                    if visited[ny, nx] or not inv[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        holes = inv & (~visited)
        return mask | holes

    def _pilBicubic(self):
        return getattr(getattr(self.Image, "Resampling", self.Image), "BICUBIC")

    def _pilNearest(self):
        return getattr(getattr(self.Image, "Resampling", self.Image), "NEAREST")


#
# Optional minimal test hook (kept small; no automated test implemented here).
#
class DefSynUSSequenceInferenceTest:
    pass
