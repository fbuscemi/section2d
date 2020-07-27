def simple_t(factory, ba, h, ta, tw, lc=None):
    # ISAMI composite T, ohne radien
    lc = min(ba, h, ta, tw) / 2.0 # crude assumption
    
    factory.addPoint(ba/2,  0,  0, lc, 1)
    factory.addPoint(ba/2,  ta, 0, lc, 2)
    factory.addPoint(tw/2,  ta, 0, lc, 3)
    factory.addPoint(tw/2,  h,  0, lc, 4)
    factory.addPoint(-tw/2, h,  0, lc, 5)
    factory.addPoint(-tw/2, ta, 0, lc, 6)
    factory.addPoint(-ba/2, ta, 0, lc, 7)
    factory.addPoint(-ba/2, 0,  0, lc, 8)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 5, 4)
    factory.addLine(5, 6, 5)
    factory.addLine(6, 7, 6)
    factory.addLine(7, 8, 7)
    factory.addLine(8, 1, 8)

    factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    factory.addPlaneSurface([1], 1)
    
    
def simple_c(factory, ba, bf, h, ta, tw, tf, lc=None):
    # ISAMI composite C, ohne radien
    lc = min(ba, bf, h, ta, tw, tf) / 2.0 # crude assumption
    
    factory.addPoint(ba,  0,  0, lc, 1)
    factory.addPoint(ba,  ta, 0, lc, 2)
    factory.addPoint(tw,  ta, 0, lc, 3)
    factory.addPoint(tw,  h-tf,  0, lc, 4)
    factory.addPoint(bf, h-tf,  0, lc, 5)
    factory.addPoint(bf, h, 0, lc, 6)
    factory.addPoint(0, h, 0, lc, 7)
    factory.addPoint(0, 0,  0, lc, 8)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 5, 4)
    factory.addLine(5, 6, 5)
    factory.addLine(6, 7, 6)
    factory.addLine(7, 8, 7)
    factory.addLine(8, 1, 8)

    factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    factory.addPlaneSurface([1], 1)
    
    
def simple_z(factory, ba, bf, h, ta, tw, tf, lc=None):
    # ISAMI composite Z, ohne radien
    lc = min(ba, bf, h, ta, tw, tf) / 2.0 # crude assumption
    
    factory.addPoint(tw/2,  0,  0, lc, 1)
    factory.addPoint(tw/2,  h-tf, 0, lc, 2)
    factory.addPoint(bf-tw/2, h-tf, 0, lc, 3)
    factory.addPoint(bf-tw/2, h,  0, lc, 4)
    factory.addPoint(-tw/2, h  0, lc, 5)
    factory.addPoint(-tw/2, ta, 0, lc, 6)
    factory.addPoint(tw/2 - ba, ta, 0, lc, 7)
    factory.addPoint(tw/2 - ta,  0, lc, 8)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 5, 4)
    factory.addLine(5, 6, 5)
    factory.addLine(6, 7, 6)
    factory.addLine(7, 8, 7)
    factory.addLine(8, 1, 8)

    factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    factory.addPlaneSurface([1], 1)
    
    
def simple_i(factory, ba, bf, h, ta, tw, tf, lc=None):
    # ISAMI composite I, ohne radien
    lc = min(ba, bf, h, ta, tw, tf) / 2.0 # crude assumption
    
    factory.addPoint(ba/2, 0,  0, lc, 1)
    factory.addPoint(ba/2, ta, 0, lc, 2)
    factory.addPoint(tw/2, ta, 0, lc, 3)
    factory.addPoint(tw/2, h-tf,  0, lc, 4)
    factory.addPoint(bf/2, h-tf  0, lc, 5)
    factory.addPoint(bf/2, h, 0, lc, 6)
    factory.addPoint(-bf/2, h, 0, lc, 7)
    factory.addPoint(-bf/2, h-tf  0, lc, 8)
    factory.addPoint(-tw/2, h-tf,  0, lc, 9)
    factory.addPoint(-tw/2, ta, 0, lc, 10)
    factory.addPoint(-ba/2, ta, 0, lc, 11)
    factory.addPoint(-ba/2, 0,  0, lc, 12)
    

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 5, 4)
    factory.addLine(5, 6, 5)
    factory.addLine(6, 7, 6)
    factory.addLine(7, 8, 7)
    factory.addLine(8, 9, 8)
    factory.addLine(9, 10, 9)
    factory.addLine(10, 11, 10)
    factory.addLine(11, 12, 11)
    factory.addLine(12, 1, 12)

    factory.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 1)
    factory.addPlaneSurface([1], 1)
    
def rect(factory, a, b, lc):
    # rectangle
    lc = min(a, b) / 6
    
    factory.addPoint(b/2, -h/2,  0, lc, 1)
    factory.addPoint(b/2, h/2, 0, lc, 2)
    factory.addPoint(-b/2, h/2, 0, lc, 3)
    factory.addPoint(-b/2, -h/2,  0, lc, 4)

    factory.addLine(1, 2, 1)
    factory.addLine(2, 3, 2)
    factory.addLine(3, 4, 3)
    factory.addLine(4, 5, 4)
    
    factory.addCurveLoop([1, 2, 3, 4], 1)
    factory.addPlaneSurface([1], 1)

    
def square(factory, a, lc):
    # rectangle
    rect(a, a)
    
    