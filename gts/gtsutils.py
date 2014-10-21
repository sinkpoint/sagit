from os import system

_SIMULATE = False
def exec_cmd(cmd):
    print '#-> '+cmd
    if not _SIMULATE:
        system(cmd)