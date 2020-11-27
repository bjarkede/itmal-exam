from data_preprocessing import *
from data_analysis import *
from helpers import *

def main(pathName, vis):
    """Try building the binary file of features. If it exists don't do anything"""
    if pathName:
        build_binaryfile(pathName)
        quit()
    if path.exists(fileName) == False:
        print("FatalError: couldn't find dataset [%s]. Use --build <path_to_data>" % fileName)
        quit()

    """Load the binary file and read it into a pandas DataFrame."""
    """Then begin data preprocessing"""
    df = load_binaryfile(fileName)
    encode_class_labels(df)

    """If --vis is set we render some of our data to disk"""
    if vis:
        perform_and_render_analysis(df)

    X_train, X_test, y_train, y_test = partion_dataset(df)
    
if __name__== "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hello, World!')
    parser.add_argument('--build', metavar='path',required=False, help='build the dataset locally')
    parser.add_argument('--vis', type=str2bool, nargs='?', const=True, default=False, help='Render images from data nalysis')
    
    args = parser.parse_args()
    main(pathName=args.build, vis=args.vis)
    
