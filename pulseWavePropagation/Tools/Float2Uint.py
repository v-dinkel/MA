
import numpy as np

def float2Uint(Image_float):
    MaxLn = np.max(Image_float)
    MinLn = np.min(Image_float)
    # LnGray = 255*(Image_float - MinLn)//(MaxLn - MinLn + 1e-6)
    LnGray = 255*((Image_float - MinLn)/float((MaxLn - MinLn + 1e-6)))

    LnGray = np.array(LnGray, dtype = np.uint8)

    return LnGray
