import numpy as np
from pathlib import Path
import pandas as pd

def xywh2xyxy(x):
    #y = x.new(x.shape)
    y = np.zeros(x.shape)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# convert xyxy to xywh
def xyxy2xywh(x):
    #y = x.new(x.shape)
    y = np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

# scale df to a given size
def scale_to_size(df, HEIGHT, WIDTH):
  df[["xmid", "w"]] *= WIDTH
  df[["ymid", "h"]] *= HEIGHT

  return df


# opposite of scale, i.e. descale by using dividing by the given size
def descale_to_size(df, HEIGHT, WIDTH):
  df[["xmid", "w"]] /= WIDTH
  df[["ymid", "h"]] /= HEIGHT

  return df


# construct dict per tile
def construct_tile_dict(row_div=3, col_div=3):
  tile_offsets = []
  for i in range(row_div):
    for j in range(col_div):
      dx, dy = (i/row_div), (j/row_div) # offsets to add
      tile_offsets.append((dx,dy))

  tile_dict = {f"tile{i+1}.txt": tile_offsets[i] for i in range(row_div*col_div)}
  return tile_dict





def stitch_labels(label_folder:str,
                  HEIGHT,
                  WIDTH, 
                  row_div=3, 
                  col_div=3):
  
  label_folder = Path(label_folder) # make str to posix path
  label_paths = label_folder.glob("*.txt") # return all .txt file in folder in posix path type
  label_paths = [path.name for path in label_paths] # get the base name of each path, e.g. path/to/tile.txt -> tile1.txt

  col_names = ["class", "xmid", "ymid", "w", "h"]

  tile_dict = construct_tile_dict(row_div=row_div, col_div=col_div)

  master_df = None # make df that will contain all labels across tiled preds

  for txt in label_paths:
    dy, dx = tile_dict[txt] # offsets to add based on tile

    df = pd.read_csv(str(label_folder / txt), names=col_names, sep=" ") # indicate that values are separated by space

    new_values = scale_to_size(df, HEIGHT/row_div, WIDTH/col_div) # scale coords by 1/3 of original height & width
    new_values = descale_to_size(df, HEIGHT, WIDTH) # decale back coords by original height & width; all coords should be bet. 0-1 after this line

    dfPrime = pd.DataFrame(data=new_values, columns=col_names[1:])
    dfPrime.insert(0, "class", df["class"].values) # insert class column

    dfPrime["xmid"] += dx
    dfPrime["ymid"] += dy

    master_df = pd.concat([dfPrime]) if master_df is None else pd.concat([master_df, dfPrime])


  return master_df 
