"""Allow ``python -m cyclist_tracker``."""
import sys

from .cli import main

sys.exit(main())
