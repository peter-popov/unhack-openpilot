# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
if sys.version_info[0] == 3:
    from . import libDlModelToolsPy3 as modeltools
    from . import libDlContainerPy3 as dlcontainer
else:
    import libDlContainerPy as dlcontainer
    import libDlModelToolsPy as modeltools
