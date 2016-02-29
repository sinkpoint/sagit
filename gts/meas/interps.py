import numpy as np

def trilinear_interp(img, indices):
    """
        trilinear interpolation of a list of float indices into an input array
        params: input_array:(IxJxK array), incides:(Nx3 indices)
        output: (N,) list of values
    """    
    input_array = np.array(img.get_data())
    indices = np.array(indices)

    x_indices = indices[:,0]
    y_indices = indices[:,1]
    z_indices = indices[:,2]

    # get lower bounds
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)

    # get upper bounds0000
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # #Check if xyz1 is beyond array boundary:
    x1[np.where(x1==input_array.shape[0])] = x0.max()
    y1[np.where(y1==input_array.shape[1])] = y0.max()
    z1[np.where(z1==input_array.shape[2])] = z0.max()

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    kx = 1 - x
    ky = 1 - y
    kz = 1 - z

    #output = input_array[x0,y0,z0]
    #print output
    output = (input_array[x0,y0,z0]*kx*ky*kz +
                 input_array[x1,y0,z0]*x*ky*kz +
                 input_array[x0,y1,z0]*kx*y*kz +
                 input_array[x0,y0,z1]*kx*ky*z +
                 input_array[x1,y0,z1]*x*ky*z +
                 input_array[x0,y1,z1]*kx*y*z +
                 input_array[x1,y1,z0]*x*y*kz +
                 input_array[x1,y1,z1]*x*y*z)

    return output