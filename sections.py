import numpy as np
import shapely
from shapely.geometry import Polygon

# baue einen bogen...
def arc(xc, yc, r, start, end, n=10, include_last=True):
    alpha = np.radians(np.linspace(start, end, n))
    vertices = np.empty((n, 2))
    vertices[:,0] = xc + r*np.cos(alpha)
    vertices[:,1] = yc + r*np.sin(alpha)
    if include_last:
        return vertices
    else:
        return vertices[:-1]

    
# these return two lists: outer contours, inner contours

def verts_rect(h, b):
    """rectangle"""
    return [np.array([(0, 0), (b, 0), (b, h), (0, h)])], []


def verts_tri(h, b):
    """symmetric triangle"""
    return [np.array([(b/2, 0), (h, 0), (-b/2, 0)])], []


def verts_circle(r):
    """solid circle"""
    return [arc(0, 0, r, 0, 360, n=72)[:-1]], []

def verts_tube(ra, ri):
    outer = arc(0, 0, ra, 0, 360, n=72)[:-1]
    inner = arc(0, 0, ri, 360, 0, n=72)[:-1]
    return [outer], [inner]

def verts_c1(h, tw, ba, ta, bf, tf):
    """c profile, no fillets"""
    vertices = np.array([
        (ba, 0),
        (ba, ta),
        (tw, ta),
        (tw, h-tf),
        (bf, h-tf),
        (bf, h),
        (0, h),
        (0, 0)
    ])
    return [vertices], []

def verts_c2(h, tw, ba, ra, bf, rf):
    """c profile, sheet metal"""
    arrays = [
        [(ba, 0), (ba, tw)],
        arc(ra+tw, ra+tw, ra, 270, 180),
        arc(rf+tw, h-tw-rf, rf, 180, 90),
        [(bf, h-tw), (bf, h)],
        arc(rf+tw, h-tw-rf, rf+tw, 90, 180),
        arc(ra+tw, ra+tw, ra+tw, 180, 270)
    ]
    return [np.concatenate(arrays)], []
        
        
def verts_z1(h, tw, ba, ta, bf, tf):
    """z profile, no fillets"""
    vertices = np.array([
        (tw, 0),
        (tw, h-tf),
        (bf, h-tf),
        (bf, h),
        (0, h),
        (0, ta),
        (tw-ba, ta),
        (tw-ba, 0)
    ])
    return [vertices], []


def verts_z2(h, tw, ba, ra, bf, rf):
    """Z section, sheet metal"""
    arrays = [
        arc(-ra, ra+tw, ra+tw, 270, 360),
        arc(rf+tw, h-tw-rf, rf, 180, 90),
        [(bf, h-tw), (bf, h)],
        arc(rf+tw, h-tw-rf, rf+tw, 90, 180),
        arc(-ra, ra+tw, ra, 0, -90),
        [(tw-ba, tw), (tw-ba, 0)],
    ]
    return [np.concatenate(arrays)], []


def verts_z02c(h, tw, ba, ra, bf, rf, blf, rlf):
    """Z section, with lipped flange, sheet metal. ISAMI Fra-Z-02-b or Stg-Z-02-c"""
    arrays = [
        arc(-ra, ra+tw, ra+tw, 270, 360),
        arc(rf+tw, h-tw-rf, rf, 180, 90),
        arc(bf-tw-rlf, h-tw-rlf, rlf, 90, 0),
        [(bf-tw, h-blf), (bf, h-blf)],
        arc(bf-tw-rlf, h-tw-rlf, rlf+tw, 0, 90),
        arc(rf+tw, h-tw-rf, rf+tw, 90, 180),
        arc(-ra, ra+tw, ra, 0, -90),
        [(tw-ba, tw), (tw-ba, 0)],
    ]
    return [np.concatenate(arrays)], []

def verts_i1(h, tw, ba, ta, bf, tf):
    """symmetric I section, no fillets"""
    vertices = np.array([
        (ba/2, 0),
        (ba/2, ta),
        (tw/2, ta),
        (tw/2, h-tf),
        (bf/2, h-tf),
        (bf/2, h),
        (-bf/2, h),
        (-bf/2, h-tf),
        (-tw/2, h-tf),
        (-tw/2, ta),
        (-ba/2, ta),
        (-ba/2, 0)
    ])
    return [vertices], []


def verts_j1(h, tw, ba, ta, bf, tf):
    """J section, no fillets"""
    vertices = np.array([
        ((ba-tw)/2, 0),
        ((ba-tw)/2, ta),
        (tw, ta),
        (tw, h-tf),
        (bf, h-tf),
        (bf, h),
        (0, h),
        (0, ta),
        ((tw-ba)/2, ta),
        ((tw-ba)/2, 0),
    ])
    return [vertices], []


def verts_l1(h, tw, ba, ta):
    """L section, no fillets"""
    vertices = np.array([
        (ba, 0),
        (ba, ta),
        (tw, ta),
        (tw, h),
        (0, h),
        (0, 0)
    ])
    return [vertices], []


def verts_l2(h, tw, ba, ra):
    """L section, sheet metal"""
    arrays = [
        [(ba, 0), (ba, tw)],
        arc(ra+tw, ra+tw, ra, 270, 180),
        [(tw, h), (0, h)],
        arc(ra+tw, ra+tw, ra+tw, 180, 270),
    ]
    return [np.concatenate(arrays)], []


VERTEX_FUNCTIONS = {
    "rect":   (verts_rect, ["h", "b"]),
    "tri":    (verts_tri, ["h", "b"]),
    "circle": (verts_circle, ["r"]),
    "tube":   (verts_tube, ["ra", "ri"]),
    "c1":     (verts_c1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "c2":     (verts_c2, ["h", "tw", "ba", "ra", "bf", "rf"]),
    "i1":     (verts_i1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "j1":     (verts_j1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "l1":     (verts_l1, ["h", "tw", "ba", "ta"]),
    "l2":     (verts_l2, ["h", "tw", "ba", "ra"]),
    "z1":     (verts_z1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "z2":     (verts_z2, ["h", "tw", "ba", "ra", "bf", "rf"]),
    "z02c":   (verts_z02c, ["h", "tw", "ba", "ra", "bf", "rf", "blf", "rlf"]),
}


def section(kind, *args):
    "Return a shapely polygon"
    # outer, inner: lists of vertices
    func, attrib_names = VERTEX_FUNCTIONS[kind]
    #x = [params[k] for k in attrib_names]
    outer, inner = func(*args)
    #try:
    p = Polygon(outer[0])
    for verts in outer[1:]:
        p = p.union(Polygon(verts))
    for verts in inner:
        p = p.difference(Polygon(verts))
    #assert p.is_valid
    return p if p.is_valid else None
    #except:
    #    return None
    
