"""polygon integrals

https://people.sc.fsu.edu/~jburkardt/py_src/polygon_integrals/polygon_integrals.html

We suppose that POLY is a planar polygon with N vertices X, Y, listed in counterclockwise order.

For nonnegative integers P and Q, the (unnormalized) moment of order (P,Q) for POLY is defined by:

        Nu(P,Q) = Integral ( x, y in POLY ) x^p y^q dx dy
      
In particular, Nu(0,0) is the area of POLY.
Simple formulas are available for low orders:

        Nu(0,0) = 1/2 (1<=i<=N) X(i-1)Y(i)-X(i)Y(i-1)
        Nu(1,0) = 1/6 (1<=i<=N) ( X(i-1)Y(i)-X(i)Y(i-1) ) * (X(i-1)+X(i))
        Nu(0,1) = 1/6 (1<=i<=N) ( X(i-1)Y(i)-X(i)Y(i-1) ) * (Y(i-1)+Y(i))
        Nu(1,1) = 1/24 (1<=i<=N) ( X(i-1)Y(i)-X(i)Y(i-1) ) * (2X(i-1)Y(i-1)+X(i-1)Y(i)+X(i)Y(i-1)+2X(i)Y(i))
        Nu(2,0) = 1/12 (1<=i<=N) ( X(i-1)Y(i)-X(i)Y(i-1) ) * (X(i-1)^2+X(i-1)X(i)+X(i)^2)
        Nu(0,2) = 1/12 (1<=i<=N) ( X(i-1)Y(i)-X(i)Y(i-1) ) * (Y(i-1)^2+Y(i-1)Y(i)+Y(i)^2)
"""

def integrate00(vertices):
    """integrate the function F(x,y) = 1 over the polygon. Same as area.
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += x[i-1]*y[i] - x[i]*y[i-1]
    return total/2.0


def integrate10(vertices):
    """integrate the function F(x,y) = x over the polygon. 
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += (x[i-1]*y[i] - x[i]*y[i-1]) * (x[i-1] + x[i])
    return total/6.0    


def integrate01(vertices):
    """integrate the function F(x,y) = y over the polygon. 
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += (x[i-1]*y[i] - x[i]*y[i-1]) * (y[i-1] + y[i])
    return total/6.0        


def integrate20(vertices):
    """integrate the function F(x,y) = x^2 over the polygon. 
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += (x[i-1]*y[i] - x[i]*y[i-1]) * (x[i-1]**2 + x[i-1]*x[i] + x[i]**2)
    return total/12.0    


def integrate11(vertices):
    """integrate the function F(x,y) = x*y over the polygon. 
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += (x[i-1]*y[i] - x[i]*y[i-1]) * \
                 (2*x[i-1]*y[i-1] + x[i-1]*y[i] + x[i]*y[i-1] + 2*x[i]*y[i])
    return total/24.0


def integrate02(vertices):
    """integrate the function F(x,y) = y^2 over the polygon. 
    vertices: float array of shape (n, 2)
    returns: float
    """
    n = len(vertices)
    x, y = vertices.T 
    total = 0.0
    for i in range(n):
        total += (x[i-1]*y[i] - x[i]*y[i-1]) * (y[i-1]**2 + y[i-1]*y[i] + y[i]**2)
    return total/12.0


def polygon_area(outers, inners):
    total_area = 0. 
    for contour in outers:
        area = integrate00(contour)
        total_area += abs(area)
    for contour in inners:
        area = integrate00(contour)
        total_area -= abs(area)        
    return total_area