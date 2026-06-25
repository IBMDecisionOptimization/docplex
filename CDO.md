Here is a **clean, production-ready `pyproject.toml` setup** for the **plugin/backend architecture** you described.

We’ll define **4 distributions**:

```
alpha                (top-level package)
beta                 (API layer)
beta-backend-lite    (limited binaries)
beta-backend-full    (unlimited binaries)
```

***

# 🧩 1. `beta` (API-only layer)

This package exposes the interface and dynamically loads a backend.

### `beta/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "beta"
version = "1.0.0"
description = "Core API for beta with pluggable backends"
authors = [{ name = "Your Name" }]
requires-python = ">=3.9"
dependencies = []

[project.optional-dependencies]
lite = ["beta-backend-lite"]
full = ["beta-backend-full"]
```

### Backend loading (inside `beta`)

```python
# beta/backend.py
import importlib.metadata


def load_backend():
    # Discover installed backend via entry points
    eps = importlib.metadata.entry_points(group="beta.backends")

    if not eps:
        raise RuntimeError("No beta backend installed")

    # Choose first available (or enhance selection logic)
    backend = eps[0].load()
    return backend
```

***

# ⚙️ 2. `beta-backend-lite`

Contains limited binaries.

### `beta-backend-lite/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "beta-backend-lite"
version = "1.0.0"
description = "Lite backend for beta (limited capabilities)"
requires-python = ">=3.9"
dependencies = ["beta"]

[project.entry-points."beta.backends"]
lite = "beta_backend_lite.backend:Backend"
```

***

# 🚀 3. `beta-backend-full`

Contains unlimited binaries.

### `beta-backend-full/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "beta-backend-full"
version = "1.0.0"
description = "Full backend for beta (unlimited capabilities)"
requires-python = ">=3.9"
dependencies = ["beta"]

[project.entry-points."beta.backends"]
full = "beta_backend_full.backend:Backend"
```

***

# 📦 4. `alpha` (top-level consumer)

Depends on `beta`, but not directly on a backend.

### `alpha/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "alpha"
version = "1.0.0"
description = "Alpha package depending on beta"
requires-python = ">=3.9"
dependencies = ["beta"]

[project.optional-dependencies]
lite = ["beta-backend-lite"]
full = ["beta-backend-full"]
```

***

# ✅ Installation UX

### Default (user picks backend explicitly)

```
pip install alpha beta-backend-lite
```

or

```
pip install alpha beta-backend-full
```

***

### Convenience via extras

```
pip install alpha[lite]
pip install alpha[full]
```

***

# ✅ Recommended runtime selection improvement

You should add **deterministic backend selection**, e.g.:

```python
def load_backend(preferred=None):
    eps = {
        ep.name: ep
        for ep in importlib.metadata.entry_points(group="beta.backends")
    }

    if preferred:
        return eps[preferred].load()

    if len(eps) == 1:
        return next(iter(eps.values())).load()

    raise RuntimeError(
        f"Multiple backends available: {list(eps.keys())}, specify one"
    )
```

And allow:

* environment variable (`BETA_BACKEND=full`)
* config file
* explicit parameter

***

# ✅ Key design advantages of this setup

* ✅ **Immutable packages** (no upgrade mutation)
* ✅ **Clear separation**
  * beta = API
  * backends = binaries
* ✅ **Extensible**
  * future: `beta-backend-gpu`, `beta-backend-cloud`
* ✅ **Clean dependency graph**
* ✅ **Professional-grade packaging (entry points)**

***

# 💡 Optional enhancements

### 1. Prevent multiple backend conflicts

You can:

* Detect multiple backends at runtime (recommended ✅)
* Or enforce via documentation

***

### 2. Platform-specific wheels

Each backend can ship:

* manylinux wheels
* Windows wheels
* macOS wheels

***

### 3. Version alignment

Ensure compatibility:

```toml
dependencies = ["beta>=1.0,<2.0"]
```

***

# ✅ Final takeaway

This architecture gives you:

✔ maximum flexibility  
✔ clean packaging semantics  
✔ no surprises for users  
✔ future extensibility

***

If you want, I can next:

* show a **repo layout (monorepo or multi-repo)**
* or add **CI/CD wheel build strategy (manylinux + cibuildwheel)**
