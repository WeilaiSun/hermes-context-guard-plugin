"""One-click installer for context-guard--plugin v2.3.

Detects Hermes installation, applies the turn_context.py patch, installs
Python dependencies, and verifies the setup.

Usage:
    python install.py                    # auto-detect HERMES_HOME
    python install.py --hermes-home /path/to/hermes
    python install.py --dry-run          # check without modifying
    python install.py --uninstall        # reverse the patch

The installer is idempotent: running it twice won't double-patch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent
PATCH_FILE = PLUGIN_DIR / "context-guard--plugin-patch.diff"
REQUIREMENTS_FILE = PLUGIN_DIR / "requirements.txt"
TARGET_FILE_REL = "hermes-agent" / "agent" / "turn_context.py"


def detect_hermes_home() -> Path | None:
    """Auto-detect HERMES_HOME from environment or common locations."""
    env_home = os.environ.get("HERMES_HOME")
    if env_home and Path(env_home).exists():
        return Path(env_home)

    candidates = [
        Path.home() / ".hermes",
        Path("F:/Hermes/HERMES_HOME"),
        Path("/f/Hermes/HERMES_HOME"),
        Path.cwd().parent,
    ]
    for c in candidates:
        target = c / TARGET_FILE_REL
        if target.exists():
            return c
    return None


def detect_hermes_version(hermes_home: Path) -> str:
    """Detect the Hermes version from the installation."""
    version_file = hermes_home / "hermes-agent" / "agent" / "__init__.py"
    if version_file.exists():
        try:
            content = version_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                if "__version__" in line:
                    return line.split("=")[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return "unknown"


def check_patch_applied(hermes_home: Path) -> bool:
    """Check if the patch has already been applied."""
    target = hermes_home / TARGET_FILE_REL
    if not target.exists():
        return False
    try:
        content = target.read_text(encoding="utf-8")
        return "_plugin_replaced_messages" in content and "context-guard--plugin" in content
    except Exception:
        return False


def apply_patch(hermes_home: Path, dry_run: bool = False) -> bool:
    """Apply the turn_context.py patch using the `patch` command."""
    if check_patch_applied(hermes_home):
        print("[OK] Patch already applied, skipping.")
        return True

    if not PATCH_FILE.exists():
        print(f"[ERROR] Patch file not found: {PATCH_FILE}")
        return False

    target = hermes_home / TARGET_FILE_REL
    if not target.exists():
        print(f"[ERROR] Target file not found: {target}")
        return False

    if dry_run:
        cmd = ["patch", "--dry-run", "-p1"]
    else:
        cmd = ["patch", "-p1", "--forward"]

    print(f"{'[DRY-RUN] ' if dry_run else ''}Applying patch to {target}...")
    result = subprocess.run(
        cmd,
        stdin=open(PATCH_FILE, "r", encoding="utf-8"),
        cwd=str(hermes_home),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("[OK] Patch applied successfully.")
        return True
    else:
        print(f"[ERROR] Patch failed: {result.stderr or result.stdout}")
        return False


def reverse_patch(hermes_home: Path) -> bool:
    """Reverse the patch (uninstall)."""
    if not check_patch_applied(hermes_home):
        print("[OK] Patch not applied, nothing to reverse.")
        return True

    cmd = ["patch", "-p1", "--reverse", "--forward"]
    print("Reversing patch...")
    result = subprocess.run(
        cmd,
        stdin=open(PATCH_FILE, "r", encoding="utf-8"),
        cwd=str(hermes_home),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("[OK] Patch reversed successfully.")
        return True
    else:
        print(f"[ERROR] Reverse failed: {result.stderr or result.stdout}")
        return False


def install_dependencies(dry_run: bool = False) -> bool:
    """Install Python dependencies from requirements.txt."""
    if not REQUIREMENTS_FILE.exists():
        print(f"[ERROR] requirements.txt not found: {REQUIREMENTS_FILE}")
        return False

    if dry_run:
        print("[DRY-RUN] Would install dependencies from requirements.txt")
        return True

    print("Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("[OK] Dependencies installed.")
        return True
    else:
        print(f"[ERROR] Dependency install failed: {result.stderr}")
        return False


def copy_plugin_files(hermes_home: Path, dry_run: bool = False) -> bool:
    """Copy plugin files to $HERMES_HOME/plugins/context-guard--plugin/."""
    dest = hermes_home / "plugins" / "context-guard--plugin"
    
    if dry_run:
        print(f"[DRY-RUN] Would copy plugin files to {dest}")
        return True
    
    print(f"Copying plugin files to {dest}...")
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(PLUGIN_DIR, dest, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", ".hermes"
        ))
        print("[OK] Plugin files copied.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to copy plugin files: {e}")
        return False


def register_plugin(hermes_home: Path, dry_run: bool = False) -> bool:
    """Register context-guard--plugin in config.yaml plugins list."""
    import yaml
    
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] config.yaml not found: {config_path}")
        return False
    
    if dry_run:
        print("[DRY-RUN] Would register plugin in config.yaml")
        return True
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        plugins = cfg.get("plugins", [])
        if plugins is None:
            plugins = []
        
        if "context-guard--plugin" in plugins:
            print("[OK] Plugin already registered in config.yaml.")
            return True
        
        plugins.append("context-guard--plugin")
        cfg["plugins"] = plugins
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        print("[OK] Plugin registered in config.yaml.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to register plugin: {e}")
        return False


def verify_installation(hermes_home: Path) -> bool:
    """Verify the plugin installation is correct."""
    target = hermes_home / TARGET_FILE_REL
    plugin_yaml = PLUGIN_DIR / "plugin.yaml"
    init_py = PLUGIN_DIR / "__init__.py"

    checks = [
        (plugin_yaml.exists(), f"plugin.yaml exists: {plugin_yaml}"),
        (init_py.exists(), f"__init__.py exists: {init_py}"),
        (target.exists(), f"turn_context.py exists: {target}"),
        (check_patch_applied(hermes_home), "Patch applied to turn_context.py"),
    ]

    all_ok = True
    for ok, msg in checks:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {msg}")
        if not ok:
            all_ok = False

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Install context-guard--plugin v2.3")
    parser.add_argument("--hermes-home", type=str, help="Path to HERMES_HOME")
    parser.add_argument("--dry-run", action="store_true", help="Check without modifying")
    parser.add_argument("--uninstall", action="store_true", help="Reverse the patch")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()

    hermes_home = Path(args.hermes_home) if args.hermes_home else detect_hermes_home()
    if hermes_home is None or not hermes_home.exists():
        print("[ERROR] Could not detect HERMES_HOME. Use --hermes-home to specify.")
        return 1

    version = detect_hermes_version(hermes_home)
    print(f"Hermes version: {version}")
    print(f"HERMES_HOME: {hermes_home}")
    print(f"Plugin dir: {PLUGIN_DIR}")
    print()

    if args.uninstall:
        if not reverse_patch(hermes_home):
            return 1
        print("\n[OK] Uninstall complete.")
        return 0

    if not apply_patch(hermes_home, dry_run=args.dry_run):
        return 1

    if not copy_plugin_files(hermes_home, dry_run=args.dry_run):
        return 1

    if not register_plugin(hermes_home, dry_run=args.dry_run):
        print("[WARN] Could not register plugin in config.yaml. Please add 'context-guard--plugin' manually.")

    if not args.no_deps and not args.dry_run:
        if not install_dependencies():
            print("[WARN] Dependency installation failed, continuing anyway.")

    print("\nVerification:")
    if verify_installation(hermes_home):
        print("\n[OK] context-guard--plugin v2.3 installed successfully!")
        return 0
    else:
        print("\n[WARN] Some checks failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
