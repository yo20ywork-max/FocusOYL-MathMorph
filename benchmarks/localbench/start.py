"""Create an isolated evaluator; never install into the user's main Python."""
from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = 'lm_eval[api,ifeval]==0.4.13'


def main() -> int:
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or later is required. Use py -3.12 start.py.')
    target = ROOT / '.venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        print('Creating isolated evaluation environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(target)
    check = [str(python), '-c', 'import importlib.metadata as m; assert m.version("lm_eval")=="0.4.13"; import langdetect, nltk']
    if subprocess.run(check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', REQUIREMENTS], check=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', DO_NOT_TRACK='1',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    # Credentials are not required. Inference goes only to an authenticated localhost process.
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'):
        env.pop(key, None)
    command = [str(python), '-u', str(ROOT / 'run_benchmark.py'), *sys.argv[1:]]
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        # The child receives console Ctrl+C too; allow its finally blocks to close the server.
        try:
            child.wait(timeout=25)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == 'nt' and child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif child.poll() is None:
                child.terminate()
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
