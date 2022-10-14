from .scst_loss import ScstRewardCriterion
from .label_smoothed_cross_entropy import AdjustLabelSmoothedCrossEntropyCriterion
from .clip_scst_loss import ClipScstRewardCriterion
from .label_smoothed_encouraging_loss import AdjustLabelSmoothedEncouragingLossCriterion

from .label_smoothed_cross_entropy_for_masked_ofa import AdjustLabelSmoothedCrossEntropyCriterionMaskedOFA

# from .patch_ofa_custom_criterion_v1 import CustomCriterionV1
# from .patch_ofa_custom_criterion_v2 import CustomCriterionV2
from .patch_ofa_custom_criterion_v3 import CustomCriterionV3
from .patch_ofa_custom_criterion_v4 import CustomCriterionV4
from .patch_ofa_custom_criterion_v4_1 import CustomCriterionV4_1
from .patch_ofa_custom_criterion_v4_2 import CustomCriterionV4_2
from .patch_ofa_custom_criterion_v4_3 import CustomCriterionV4_3

from .seg_criterion_v1 import SegCriterionV1
from .seg_criterion_v2 import SegCriterionV2
from .seg_criterion_v3 import SegCriterionV3
from .seg_criterion_v4 import SegCriterionV4
from .seg_criterion_mlp import SegCriterionMLP