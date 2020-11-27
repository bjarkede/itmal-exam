from data_preprocessing import *

def main():
    """Try building the binary file of features. If it exists don't do anything"""
    build_binaryfile()

    """Load the binary file and read it into a pandas DataFrame."""
    """Then begin data preprocessing"""
    df = load_binaryfile(fileName)
    encode_class_labels(df)

    X_train, X_test, y_train, y_test = partion_dataset(df)
    

if __name__== "__main__":
    main()
    
