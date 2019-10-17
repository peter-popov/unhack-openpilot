# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# @deprecated
# to allow for backward compatibility adding this import also at top-level so that
# it is possible to do <from snpe import modeltools>
# moving forward the signature to use will be <from snpe.dlc_utils import modeltools>
import sys
try:
    if sys.version_info[0] == 3:
        from snpe.dlc_utils import libDlModelToolsPy3 as modeltools
        from snpe.dlc_utils import libDlContainerPy3 as dlcontainer
    else:
        from snpe.dlc_utils import libDlContainerPy as dlcontainer
        from snpe.dlc_utils import libDlModelToolsPy as modeltools
except:
    pass
