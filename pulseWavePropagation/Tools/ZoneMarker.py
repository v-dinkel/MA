
import cv2

def zoneMarker(Img, discRegionParameter):

    discCenter =discRegionParameter['discCenter']
    discRadius =discRegionParameter['discRadius']

    cv2.circle(Img, center=(discCenter[1], discCenter[0]), radius=discRadius, color=(255, 255, 255), thickness=5)
    cv2.circle(Img, center=(discCenter[1], discCenter[0]), radius=2 * discRadius, color=(255, 255, 255),
               thickness=2)  # 2-3 RegionB
    cv2.circle(Img, center=(discCenter[1], discCenter[0]), radius=3 * discRadius, color=(255, 255, 255),
               thickness=2)
    cv2.putText(Img, "Zone B", (discCenter[1] - 50, discCenter[0] - int(2.2 * discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(Img, center=(discCenter[1], discCenter[0]), radius=5 * discRadius, color=(255, 255, 255),
               thickness=2)  # 3-5 REgionC
    cv2.putText(Img, "Zone C", (discCenter[1] - 50, discCenter[0] - int(4 * discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    return Img
