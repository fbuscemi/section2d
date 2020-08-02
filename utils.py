#import matplotlib
#matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import cv2

BLACK = (0,0,0)
WHITE = (255,255,255)

def image_cv2(outers, inners, figsize=300, pad=20):
    """Return np.array. background is white (1), shape black (0)"""

    mins = np.min(np.concatenate(outers), axis=0)
    maxs = np.max(np.concatenate(outers), axis=0)

    cx = (mins[0] + maxs[0])/2
    cy = (mins[1] + maxs[1])/2
    center = [cx, cy] # in true coordinates

    w = maxs[0] - mins[0]
    h = maxs[1] - mins[1]
    scale = (figsize - 2*pad) / max(w, h)
    
    offset = (figsize//2, figsize//2) # in image coords

    img = 255*np.ones((figsize, figsize, 3), dtype = "uint8") # empty white image

    polys = [(scale * (p - center)).astype(np.int32) for p in outers]
    polys = [p*[1,-1] for p in polys] # flip y
    cv2.fillPoly(img, polys, BLACK, cv2.LINE_AA, 0, offset)

    polys = [(scale * (p - center)).astype(np.int32) for p in inners]
    polys = [p*[1,-1] for p in polys] # flip y
    cv2.fillPoly(img, polys, WHITE, cv2.LINE_AA, 0, offset)
    
    return img


def image_mpl(outers, inners, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(1,1))
        
    for outer in outers:
        mppoly = matplotlib.patches.Polygon(outer, facecolor='k', linewidth=0)
        ax.add_patch(mppoly)
    for inner in inners:
        mppoly = matplotlib.patches.Polygon(inner, facecolor='w', linewidth=0)
        ax.add_patch(mppoly)

    ax.autoscale(tight=True)
    ax.set_aspect("equal")
    ax.axis("off")
    
    return ax.figure