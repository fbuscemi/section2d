def conv_dims(input_size, k, s, p):
    """Return output size for a 2D convolution kernel (square input, 
    square kernel). See https://arxiv.org/pdf/1603.07285v2.pdf, relationship 6

    Parameters
    ----------
    input_size : int
        size (width) of input feature
    k : int
        convolution kernel size
    s : int
        convolution stride
    p : int 
        zero padding

    Returns
    -------
    output_size : int
        size (width) of output feature
    
    """
    return (input_size + 2*p - k)//s + 1



def pool_dims(input_size, k, s):
    """Return output size for a 2D convolution kernel (square input, 
    square kernel). See https://arxiv.org/pdf/1603.07285v2.pdf, relationship 7

    Parameters
    ----------
    input_size : int
        size (width) of input feature
    k : int
        pooling kernel size
    s : int
        pooling stride

    Returns
    -------
    output_size : int
        size (width) of output feature
    
    """
    return (input_size - k)//s + 1