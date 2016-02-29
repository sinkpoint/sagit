from os import system
from subprocess import Popen, PIPE
import shlex,sys

_SIMULATE = False
def exec_cmd(cmd, truncate=False, display=True, watch='stdout', dryrun=False):
    def output_line(line, truncate=False):
        if truncate:
            line = line.replace('\n','')             
            sys.stdout.write('\b'*stdout_last_len)
            sys.stdout.write(' '*stdout_last_len)
            sys.stdout.write('\b'*stdout_last_len)

        sys.stdout.write(line)
        sys.stdout.flush()
        return len(line)

    print '#-> '+cmd
    if not _SIMULATE and not dryrun:
        #system(cmd)
        args = shlex.split(cmd)
        if watch=='stdout':
            p = Popen(args, stdout=PIPE, bufsize=1)            
            stream = p.stdout
        else:
            p = Popen(args, stderr=PIPE, bufsize=1)            
            stream = p.stderr        

        if display:
            stdout_last_len=0
            for line in iter(stream.readline, b''):                   
                stdout_last_len=output_line(line, truncate)      
        p.communicate()
        sys.stdout.write(' '*80+'\n')

