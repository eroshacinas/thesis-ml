import os
from pathlib import Path
import cv2

def tile_img(img, save_path:str, 
             row_div=3, 
             col_div=3):
  
  
  HEIGHT, WIDTH = img.shape[:2]

  save_path = Path(save_path) # make str to posix path

  if not save_path.exists():
    os.makedirs(save_path, exist_ok=True)
  
  for i in range(row_div):
    for j in range(col_div):
      
      tile = img[int(i/row_div * HEIGHT) : int((i+1)/row_div * HEIGHT), int(j/col_div * WIDTH) : int((j+1)/col_div * WIDTH)]
      
      name = "tile" + str(row_div*i + j+1) + ".jpg"
      cv2.imwrite(str(save_path / name), tile)
