
import cv2
import numpy as np

"""
This program gets the masks of different regions, eg. Region B, Region C
"""

def diskMask(Skeleton, diskCenter, diskRadius, ):
    Skeleton[Skeleton>0]=1

    ##Mask the Disk out Only

    height, width = Skeleton.shape[:2]

    DiskMask = np.ones((height, width))
    cv2.circle(DiskMask, (diskCenter[1], diskCenter[0]),  diskRadius, 0, -1)
    DiskMask = np.uint8(DiskMask)

    VesselSkeleton_nonDisc = cv2.bitwise_and(Skeleton, Skeleton, mask=DiskMask)

    return DiskMask, VesselSkeleton_nonDisc


def regionBMask(Skeleton, diskCenter, diskRadius, factor_B=(3, 1)):
    # type: (object, object, object, object) -> object
    # factor_B = [3, 2]
    # factor_C = [5, 2]

    factorB = (np.maximum(factor_B[0], factor_B[1]), np.minimum(factor_B[0], factor_B[1]))

    height, width = Skeleton.shape[:2]

    DiskMask_B = np.zeros((height, width))
    cv2.circle(DiskMask_B, (diskCenter[1], diskCenter[0]), int(factorB[0] * diskRadius), 1, -1)
    cv2.circle(DiskMask_B, (diskCenter[1], diskCenter[0]), int(factorB[1] * diskRadius), 0, -1)
    DiskMask_B = np.uint8(DiskMask_B)
    VesselSkeleton_B = cv2.bitwise_and(Skeleton, Skeleton, mask=DiskMask_B)

    return DiskMask_B, VesselSkeleton_B


def regionCMask(Skeleton, diskCenter, diskRadius, factor_C=(8, 1)):
    # factor_B = [3, 2]
    # factor_C = [5, 2]

    factorC = (np.maximum(factor_C[0], factor_C[1]), np.minimum(factor_C[0], factor_C[1]))

    height, width = Skeleton.shape[:2]

    DiskMask_C = np.zeros((height, width))
    cv2.circle(DiskMask_C, (diskCenter[1], diskCenter[0]), int(factorC[0] * diskRadius), 1, -1)
    cv2.circle(DiskMask_C, (diskCenter[1], diskCenter[0]), int(factorC[1] * diskRadius), 0, -1)
    DiskMask_C = np.uint8(DiskMask_C)

    VesselSkeleton_C = cv2.bitwise_and(Skeleton, Skeleton, mask=DiskMask_C)

    return DiskMask_C, VesselSkeleton_C
