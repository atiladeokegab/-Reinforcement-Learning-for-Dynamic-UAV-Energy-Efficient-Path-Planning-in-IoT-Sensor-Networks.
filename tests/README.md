## Testing

### Run Unit Tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html

# Discover and run all tests
pytest

# Verbose output (show each test name)
pytest -v

# Show print statements
pytest -s

# Stop at first failure
pytest -x

# Run tests matching pattern
pytest -k "sensor"

# Run specific file
pytest tests/test_iot_sensors.py

# Run with coverage
pytest --cov=src

# List tests without running
pytest --collect-only

# Run last failed tests
pytest --lf

# Run in parallel (faster)
pytest -n auto  # Requires: pip install pytest-xdist
```

### Test Structure
```
tests/
├── __init__.py
├── test_iot_sensors.py       # IoT sensor tests (47 tests)
├── test_uav.py               # UAV tests
├── test_environment.py       # Environment tests
└── test_q_learning_agent.py  # Agent tests
```

Use WSL Browser Launcher

bash# Use wslview (if available)
```
wslview htmlcov/index.html

# Or set BROWSER variable
export BROWSER=wslview
pytest --cov=src --cov-report=html
open htmlcov/index.html  # Now works
```