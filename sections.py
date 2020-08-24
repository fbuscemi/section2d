import numpy as np
import shapely
from shapely.geometry import Polygon
from poly import integrate00
from utils import image_cv2, image_mpl

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
    """circular tube"""
    outer = arc(0, 0, ra, 0, 360, n=72)[:-1]
    inner = arc(0, 0, ri, 360, 0, n=72)[:-1]
    return [outer], [inner]

def verts_c1(h, tw, ba, ta, bf, tf):
    """c profile, no fillets. 
    
    Corresponds to:
    - Metal_Stg_C_01_b
    - Metal_Fra_C_01_c
    - Metal_Stg_L_02_b (bf -> blw, tf -> blw)
    """
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
    """c profile, sheet metal
    
    Corrensponds to: 
    - Metal_Stg_C_01_c
    - Metal_Stg_L_02_c (bf -> blw)
    - Metal_Fra_C_05_b
    """
    arrays = [
        [(ba, 0), (ba, tw)],
        arc(ra+tw, ra+tw, ra, 270, 180),
        arc(rf+tw, h-tw-rf, rf, 180, 90),
        [(bf, h-tw), (bf, h)],
        arc(rf+tw, h-tw-rf, rf+tw, 90, 180),
        arc(ra+tw, ra+tw, ra+tw, 180, 270)
    ]
    return [np.concatenate(arrays)], []
        
def verts_c3(h, tw, ba, ta, ra, bf, tf, rf):
    """c profile, with fillets, machined or extruded
    
    Corrensponds to: 
    - Metal_Stg_C_01_a
    - Metal_Fra_C_01_c
    - Metal_Stg_L_02_a (blw -> bf, tlf -> tf, rlw -> rf)
    """
    arrays = [
        [(ba, 0), 
        (ba, tw)],
        arc(ra+tw, ra+ta, ra, 270, 180),
        arc(rf+tw, h-tf-rf, rf, 180, 90),
        [(bf, h-tf), (bf, h), (0, h), (0, 0)]
    ]
    return [np.concatenate(arrays)], []
        

def verts_e1(h, hs, twa, twf, ba, ta, bs, ts, bf, tf):
    """e profile, no fillets.
    
    Corresponds to:
    - Metal_Fra_E_01_d
    """
    vertices = np.array([
        (ba, 0),
        (ba, ta),
        (twa, ta),
        (twa, hs),
        (ts, hs),
        (ts, hs+bs),
        (twf, hs+bs),
        (twf, h-tf),
        (bf, h-tf),
        (bf, h),
        (0, h),
        (0, 0)
    ])
    return [vertices], []

        
def verts_z1(h, tw, ba, ta, bf, tf):
    """z profile, no fillets.
    
    Corresponds to:
    - Metal_Stg_Z_01_b
    """
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
    """Z section, sheet metal
    
    Corresponds to:
    - Metal_Stg_Z_01_c
    """
    arrays = [
        arc(-ra, ra+tw, ra+tw, 270, 360),
        arc(rf+tw, h-tw-rf, rf, 180, 90),
        [(bf, h-tw), (bf, h)],
        arc(rf+tw, h-tw-rf, rf+tw, 90, 180),
        arc(-ra, ra+tw, ra, 0, -90),
        [(tw-ba, tw), (tw-ba, 0)],
    ]
    return [np.concatenate(arrays)], []


def verts_z3(h, tw, ba, ra, bf, rf, blf, rlf):
    """Z section, with one lipped flange, sheet metal.

    Corresponds to:
    - Metal_Fra_Z_02_b
    - Metal_Stg_Z_02_c
    """
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


def verts_z4(h, tw, ba, ta, bf, tf, blf, tlf):
    """Z section, with one lipped flange, no fillets.

    Corresponds to:
    - Metal_Stg_Z_02_b
    """
    vertices = np.array([
        (tw, 0),
        (tw, h-tf),
        (bf-tlf, h-tf),
        (bf-tlf, h-blf),
        (bf, h-blf),
        (bf, h),
        (0, h),
        (0, ta),
        (tw-ba, ta),
        (tw-ba, 0)
    ])
    return [vertices], []


def verts_z5(h, tw, ba, ta, ra, bf, tf, rf):
    """Z section, with fillets.
    
    Corresponds to:
    - Metal_Stg_Z_01_a
    """
    arrays = [
        arc(rf+tw, h-tf-rf, rf, 180, 90),
        [(bf, h-tf), (bf, h), (0, h)],
        arc(-ra, ta+ra, ra, 360, 270),
        [(tw-ba, ta), (tw-ba, 0), (tw, 0)]
    ]
    return [np.concatenate(arrays)], []


def verts_z6(h, tw, ba, ta, rf):
    """Z section, round flange, no fillets.
    
    Corresponds to:
    - Metal_Stg_Z_04_b
    """
    arrays = [
        # use more points than usual on large round flange
        arc(rf, h-rf, rf-tw, 180, 0, n=40),
        arc(rf, h-rf, rf, 0, 180, n=40),
        [(0, ta), (tw-ba, ta), (tw-ba, 0), (0, 0)]
    ]
    return [np.concatenate(arrays)], []


def verts_z8(h, tw, ba, ta, ra, rf):
    """Z section, round flange, fillets.
    
    Corresponds to:
    - Metal_Stg_Z_04_a
    """
    arrays = [
        # use more points than usual on large round flange
        arc(rf, h-rf, rf-tw, 180, 0, n=80),
        arc(rf, h-rf, rf, 0, 180, n=80),
        arc(-ra, ta+ra, ra, 360, 270),
        [(tw-ba, ta), (tw-ba, 0), (tw, 0)]
    ]
    return [np.concatenate(arrays)], []


def verts_z9(h, tw, ba, ta, ra, bf, tf, rf, blf, tlf, rlf):
    """Z section, with one lipped flange, machined, with fillets.

    Corresponds to:
    - Metal_Stg_Z_02_a
    """
    arrays = [
        arc(rf+tw, h-tf-rf, rf, 180, 90),
        arc(bf-tlf-rlf, h-tf-rlf, rlf, 90, 0),
        [(bf-tlf, h-blf), (bf, h-blf), (bf, h), (0, h)],
        arc(-ra, ra+ta, ra, 0, -90),
        [(tw-ba, ta), (tw-ba, 0), (tw, 0)]
    ]
    return [np.concatenate(arrays)], []


def verts_i1(h, tw, ba, ta, bf, tf):
    """symmetric I section, no fillets
    
    Corrensponds to: 
    - Metal_Stg_I_01_b
    - Metal_Fra_I_01_d
    """
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
    """J section, no fillets.
    
    Corresponds to:
    - Metal_Stg_J_01_b
    """
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


def verts_j2(h, tw, ba, ta, ra, bf, tf, rf):
    """J section, with fillets.
    
    Corresponds to:
    - Metal_Stg_J_01_a
    """
    arrays = [
        [((ba-tw)/2, 0), ((ba-tw)/2, ta)],
        arc(ra+tw, ra+ta, ra, 270, 180),
        arc(rf+tw, h-tf-rf, rf, 180, 90),
        [(bf, h-tf), (bf, h), (0, h)],
        arc(-ra, ta+ra, 360, 270),
        [((tw-ba)/2, ta), ((tw-ba)/2, 0)],
    ]
    return [np.concatenate(arrays)], []


def verts_t1(h, tw, ba, ta):
    """T section, no fillets.
    
    Corresponds to:
    - Metal_Stg_T_01_b
    """
    vertices = np.array([
        ((ba-tw)/2, 0),
        ((ba-tw)/2, ta),
        (tw, ta),
        (tw, h),
        (0, h),
        (0, ta),
        ((tw-ba)/2, ta),
        ((tw-ba)/2, 0),
    ])
    return [vertices], []


def verts_t2(h, tw, ba, ta, ra):
    """T section, with fillets.
    
    Corresponds to:
    - Metal_Stg_T_01_a
    """
    arrays = [
        [((ba-tw)/2, 0), ((ba-tw)/2, ta)],
        arc(ra+tw, ra+ta, ra, 270, 180),
        [(tw, h), (0, h)],
        arc(-ra, ta+ra, 360, 270),
        [((tw-ba)/2, ta), ((tw-ba)/2, 0)],
    ]
    return [np.concatenate(arrays)], []


def verts_l1(h, tw, ba, ta):
    """L section, no fillets
    
    Corrensponds to: 
    - Metal_Stg_L_01_b
    """
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
    """L section, sheet metal

    Corresponds to:
    - Metal_Stg_L_01_c
    """
    arrays = [
        [(ba, 0), (ba, tw)],
        arc(ra+tw, ra+tw, ra, 270, 180),
        [(tw, h), (0, h)],
        arc(ra+tw, ra+tw, ra+tw, 180, 270),
    ]
    return [np.concatenate(arrays)], []


def verts_l3(h, tw, ba, ta, ra):
    """L section, with fillets

    Corresponds to:
    - Metal_Stg_L_01_a
    """
    arrays = [
        [(ba, 0), (ba, ta)],
        arc(ra+tw, ra+ta, ra, 270, 180),
        [(tw, h), (0, h), (0, 0)]
    ]
    return [np.concatenate(arrays)], []    


VERTEX_FUNCTIONS = {
    "rect":   (verts_rect, ["h", "b"]),
    "tri":    (verts_tri, ["h", "b"]),
    "circle": (verts_circle, ["r"]),
    "tube":   (verts_tube, ["ra", "ri"]),
    "c1":     (verts_c1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "c2":     (verts_c2, ["h", "tw", "ba", "ra", "bf", "rf"]),
    "c3":     (verts_c3, ["h", "tw", "ba", "ta", "ra", "bf", "tf", "rf"]),
    "e1":     (verts_e1, ["h", "hs", "twa", "twf", "ba", "ta", "bs", "ts", "bf", "tf"]),
    "i1":     (verts_i1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "j1":     (verts_j1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "j2":     (verts_j2, ["h", "tw", "ba", "ta", "ra", "bf", "tf", "rf"]),
    "l1":     (verts_l1, ["h", "tw", "ba", "ta"]),
    "l2":     (verts_l2, ["h", "tw", "ba", "ra"]),
    "l3":     (verts_l3, ["h", "tw", "ba", "ta", "ra"]),
    "t1":     (verts_t1, ["h", "tw", "ba", "ta"]),
    "t2":     (verts_t2, ["h", "tw", "ba", "ta", "ra"]),
    "z1":     (verts_z1, ["h", "tw", "ba", "ta", "bf", "tf"]),
    "z2":     (verts_z2, ["h", "tw", "ba", "ra", "bf", "rf"]),
    "z3":     (verts_z3, ["h", "tw", "ba", "ra", "bf", "rf", "blf", "rlf"]),
    "z4":     (verts_z4, ["h", "tw", "ba", "ta", "bf", "tf", "blf", "tlf"]),
    "z5":     (verts_z5, ["h", "tw", "ba", "ta", "ra", "bf", "tf", "rf"]),
    "z6":     (verts_z6, ["h", "tw", "ba", "ta", "rf"]),
    "z8":     (verts_z8, ["h", "tw", "ba", "ta", "ra", "rf"]),
    "z9":     (verts_z9, ["h", "tw", "ba", "ta", "ra", "bf", "tf", "rf", "blf", "tlf", "rlf"]),
}


def section(kind, *args):
    "Return a shapely polygon"
    func, attrib_names = VERTEX_FUNCTIONS[kind]
    # outer, inner: lists of vertices
    outer, inner = func(*args)
    #try:
    p = Polygon(outer[0])
    for verts in outer[1:]:
        p = p.union(Polygon(verts))
    for verts in inner:
        p = p.difference(Polygon(verts))
    #assert p.is_valid
    return p if p.is_valid else None


def rotate_section(outer, inners, alpha, refpoint=(0,0)):
    pass



class Section(object):

    def __init__(self, *args):
        """we should be able to initialise this in different ways:
        - list of parameters (order as in parameter names)
        - keyword arguments
        - dictionary
        """
        #print(self.shape_type)
        #print(self.parameter_names)
        self.params = args
        #print(args)


        vertex_function = VERTEX_FUNCTIONS[self.shape_type][0]
        self.outers, self.inners = vertex_function(*self.params)


    def params(self):
        """Return list of parameters"""
        

    def params_dict(self):
        """Return dictionary of parameters"""
        pass    
    
    def to_shapely(self):
        """Return as shapely Polygon"""
        p = Polygon(self.outers[0])
        for verts in self.outers[1:]:
            p = p.union(Polygon(verts))
        for verts in self.inners:
            p = p.difference(Polygon(verts))
        return p if p.is_valid else None

    def to_mesh(self):
        return NotImplemented

    def to_figure(self, ax=None):
        """Return matplotlib.Figure"""
        return image_mpl(self.outers, self.inners, ax=ax)

    def to_PIL(self):
        """Return PIL.Image"""
        return NotImplemented

    def to_image(self, figsize=300, pad=10):
        """Return image as np.adarray (opencv etc."""
        return image_cv2(self.outers, self.inners, figsize=figsize, pad=pad)

    def area(self):
        area_ = 0
        for vertices in self.outers:
            area_ += abs(integrate00(vertices))
        for vertices in self.inners:
            area_ -= abs(integrate00(vertices))
        return area_

    def centroid(self):
        return NotImplemented

    def principal_axes(self):
        return NotImplemented

    def inertia_tensor(self, refpoint="centroid", axes="uv"):
        return NotImplemented




class SectionRect(Section):
    shape_type = "rect"
    parameter_names = VERTEX_FUNCTIONS["rect"][1]
    
class SectionC1(Section):
    shape_type = "c1"
    parameter_names = VERTEX_FUNCTIONS["c1"][1]

class SectionC2(Section):
    shape_type = "c2"
    parameter_names = VERTEX_FUNCTIONS["c2"][1]    

class SectionC3(Section):
    shape_type = "c3"
    parameter_names = VERTEX_FUNCTIONS["c3"][1]    

class SectionE1(Section):
    shape_type = "e1"
    parameter_names = VERTEX_FUNCTIONS["e1"][1]    

class SectionI1(Section):
    shape_type = "i1"
    parameter_names = VERTEX_FUNCTIONS["i1"][1]    

class SectionJ1(Section):
    shape_type = "j1"
    parameter_names = VERTEX_FUNCTIONS["j1"][1]    

class SectionJ2(Section):
    shape_type = "j2"
    parameter_names = VERTEX_FUNCTIONS["j2"][1]    
