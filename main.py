
import yaml

from PIL import Image
from os import path as opath

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'png'

def main(case: int):

    img = Image.open(opath.join(ROOT_IN, f'{case:02}.{EXT}'))
    with open(opath.join(ROOT_IN, f'{case:02}.yaml')) as f:
        params = yaml.safe_load(f)
    

if __name__ == '__main__':

    case = 0
    while(True):
        case += 1
        try:
            main(case)
        except FileNotFoundError:
            break
