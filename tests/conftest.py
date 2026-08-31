"""Test-wide setup, applied before any test module imports the application.

One job: switch off the live rainfall fetch. `app.config` reads the environment once at
import time, and pytest loads this file before it loads a test module, so setting the
variable here is what makes it take effect.

No test in this repository should depend on the network being up. A suite that goes red
because a weather service is rate-limiting is a suite nobody trusts, and the fallback path
this forces is the one that has to keep working anyway. `tests/test_rainfall.py` exercises
the live provider against a recorded payload instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("POND_RAINFALL_ENABLED", "false")
