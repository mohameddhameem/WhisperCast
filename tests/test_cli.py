import subprocess
import pytest
from whispercast import __version__

def test_cli_version():
    """Test if `whispercast --version` returns the correct version."""
    result = subprocess.run(["whispercast", "--version"], capture_output=True, text=True, check=True)
    assert __version__ in result.stdout
