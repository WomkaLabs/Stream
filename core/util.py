from assertpy import assert_that
import glob as glob_
import logging
import numexpr
import os
import pickle
import subprocess

numexpr.set_num_threads(numexpr.detect_number_of_cores())

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("boolean value expected")

def int_list_str(int_list_str: str) -> list[int]:
    return [int(int_str) for int_str in int_list_str.split(",")]

def escape_shell_command(cmd: str):
    return 

def exec_with_log(cmd: str, on_line_feed=logging.debug, expected_exit_code: int=0):
    cmd = cmd.replace("(", r"\(").replace(")", r"\)") # Escape command string for shell-version
    logging.debug(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    last_line = None
    with p.stdout:
        for last_line in iter(p.stdout.readline, b""): # b"\n"-separated lines
            on_line_feed(last_line)
        if last_line:
            last_line = last_line.decode("utf-8").replace("\n", "")
    exit_code = p.wait()
    assert_that(exit_code).is_equal_to(expected_exit_code)
    return last_line

def glob(glob_pattern="*"):
    child_paths = sorted(glob_.glob(glob_pattern))
    return [(child_path, os.path.basename(child_path)) for child_path in child_paths]

def serialize(file_path, obj):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def deserialize(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def get_commit_id():
    return exec_with_log("git rev-parse HEAD")[:7]
