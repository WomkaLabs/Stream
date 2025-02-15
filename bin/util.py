import sys
import argparse

from core.util import *

sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(format="%(message)s", handlers=[sh], level=logging.INFO)

dummy_parser = argparse.ArgumentParser()
