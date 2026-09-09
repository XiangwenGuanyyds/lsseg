from .custom_roi_heads import CustomRoIHeads
from .residual_mask_refinement import ResidualMaskRefinementHead
from .gated_residual_refinement import GatedResidualRefinementHead
from . import box_nms_mods    # noqa: F401  triggers @BOX_NMS decorators
from . import box_reg_losses  # noqa: F401  triggers @BOX_REG_LOSSES decorators
from . import box_overlap_refinement  # noqa: F401  triggers @MODELS decorators
from . import roi_relation_refinement  # noqa: F401  triggers @MODELS decorators

__all__ = ["CustomRoIHeads"]
