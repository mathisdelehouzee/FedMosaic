import pandas as pd

def load_file(path):
    """
    Loads a .npy file from the given path and returns the numpy array.
    
    Parameters:
    path (str): The path to the .npy file.
    
    Returns:
    np.ndarray: The loaded numpy array.
    """
    return pd.read_csv(path)


df = load_file("./embeddings/embeddings/fused_clip_embeddings.csv")
print(df.columns)
for column in df.columns:
    if column == 'clip_textimg_512':
        print("yipiee")

print(df.shape)

