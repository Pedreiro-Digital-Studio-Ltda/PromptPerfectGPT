import os
import sys
import subprocess

def run_pip(args):
    return subprocess.check_call([sys.executable, "-m", "pip"] + args)

def main():
    # instala requirements do próprio diretório
    here = os.path.dirname(os.path.abspath(__file__))
    req = os.path.join(here, "requirements.txt")
    if os.path.exists(req):
        run_pip(["install", "-r", req, "-U"])

if __name__ == "__main__":
    main()