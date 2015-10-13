from gts.meas import utils

if __name__ == '__main__':
    import sys
    from optparse import OptionParser

    parser = OptionParser(usage="Usage: %prog -i input")
    parser.add_option("-m", "--input_image", dest="image",help="Input nifti volume file")
    parser.add_option("-i", "--fiber", dest="fiber",help="Input vtk fiber file")
    parser.add_option("-o", "--output", dest="output",help="Ouput vtk file")
    parser.add_option("-n", "--name", dest="name", default='Scalar', help="Name of the scalar to embed, default is 'Scalar'")

    (options, args) =  parser.parse_args()

    if not options.image or not options.fiber or not options.output:
        parser.print_help()
        sys.exit(2)
    else:
        utils.image_to_vtk(options.image, options.fiber, options.output, scalar_name=options.name)