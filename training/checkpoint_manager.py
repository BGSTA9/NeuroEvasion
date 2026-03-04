"""
training/checkpoint_manager.py — Production-grade checkpoint management.

DESIGN GOALS:
    1. Atomic writes:   Write to a .tmp file first, then os.replace() to make the
                        swap appear instantaneous to the OS. A crash mid-write can
                        never corrupt the previous good checkpoint.
    2. Manifest index:  A single manifest.json always points to the latest valid
                        checkpoint. Loading is O(1) — no directory scanning needed.
    3. Rolling window:  Keep only the most recent N checkpoints to save disk space.
    4. Drive sync:      Optionally copy each checkpoint to a Google Drive mount so
                        Colab preemptions never cause data loss.
    5. Full state:      Saves not just weights but optimizer state, epsilon schedule
                        progress, win counters, and the log directory path so the
                        logger can reattach to the same TensorBoard run.

CHECKPOINT DIRECTORY LAYOUT:
    checkpoints/
        manifest.json               ← index of all valid checkpoints
        ep_001000/
            snake.pt                ← full DQNAgent state dict
            bait.pt                 ← full DQNAgent state dict
            training_state.json     ← episode, win counters, timestamps
        ep_002000/
            ...
        emergency/
            snake.pt                ← overwritten on every SIGTERM/SIGINT
            bait.pt
            training_state.json
"""

import os
import json
import time
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from agents.dqn_agent import DQNAgent

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages serialisation of full training state to disk.

    Args:
        checkpoint_dir:         Root directory for all checkpoints.
        keep_last_n:            Number of recent checkpoints to retain.
                                Older ones are pruned automatically.
        save_optimizer:         Include optimizer state in snapshots.
        atomic_write:           Use .tmp → rename pattern for crash-safety.
        drive_sync_dir:         If non-empty, mirrors each checkpoint to this
                                path (e.g. a Google Drive mount in Colab).
    """

    MANIFEST_FILE = "manifest.json"
    EMERGENCY_SUBDIR = "emergency"

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 5,
        save_optimizer: bool = True,
        atomic_write: bool = True,
        drive_sync_dir: str = "",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.atomic_write = atomic_write
        self.drive_sync_dir = Path(drive_sync_dir) if drive_sync_dir else None

        # Ensure root directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest or initialise a fresh one
        self._manifest: Dict[str, Any] = self._load_manifest()

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    def save(
        self,
        episode: int,
        snake_agent: "DQNAgent",
        bait_agent: "DQNAgent",
        training_meta: Dict[str, Any],
        subdir: Optional[str] = None,
    ) -> Path:
        """
        Persist a full checkpoint snapshot.

        Args:
            episode:        Current episode number.
            snake_agent:    The snake DQNAgent instance.
            bait_agent:     The bait DQNAgent instance.
            training_meta:  Dict with rolling counters, log_dir, etc.
            subdir:         Override the auto-generated subdirectory name.
                            Used for emergency saves ("emergency").

        Returns:
            Path to the saved checkpoint directory.
        """
        if subdir is None:
            subdir = f"ep_{episode:07d}"

        save_path = self.checkpoint_dir / subdir
        save_path.mkdir(parents=True, exist_ok=True)

        # Build the full training state payload
        state = {
            **training_meta,
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "subdir": subdir,
        }

        # --- Save agent states ---
        self._save_tensor(
            save_path / "snake.pt",
            snake_agent.get_full_state(include_optimizer=self.save_optimizer),
        )
        self._save_tensor(
            save_path / "bait.pt",
            bait_agent.get_full_state(include_optimizer=self.save_optimizer),
        )

        # --- Save training metadata ---
        self._save_json(save_path / "training_state.json", state)

        # --- Update manifest ---
        if subdir != self.EMERGENCY_SUBDIR:
            self._manifest.setdefault("checkpoints", [])
            # Avoid duplicate entries for the same subdir
            self._manifest["checkpoints"] = [
                c for c in self._manifest["checkpoints"] if c["subdir"] != subdir
            ]
            self._manifest["checkpoints"].append(
                {"episode": episode, "subdir": subdir, "timestamp": state["timestamp"]}
            )
            self._manifest["latest"] = subdir
            self._manifest["latest_episode"] = episode
            self._write_manifest()
            self._prune_old()

        # --- Optional: mirror to Google Drive ---
        if self.drive_sync_dir:
            self._sync_to_drive(save_path, subdir)

        logger.debug("Checkpoint saved → %s (episode %d)", save_path, episode)
        return save_path

    def save_emergency(
        self,
        episode: int,
        snake_agent: "DQNAgent",
        bait_agent: "DQNAgent",
        training_meta: Dict[str, Any],
    ) -> Path:
        """
        Save an emergency checkpoint, overwriting the previous emergency slot.

        This is called from the SIGTERM / SIGINT handler, so it should be as
        fast and robust as possible: no manifest updates, no pruning, no Drive sync.
        """
        save_path = self.checkpoint_dir / self.EMERGENCY_SUBDIR
        save_path.mkdir(parents=True, exist_ok=True)

        state = {
            **training_meta,
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "subdir": self.EMERGENCY_SUBDIR,
            "emergency": True,
        }

        # Write directly (no atomic wrapper — speed matters here)
        torch.save(
            snake_agent.get_full_state(include_optimizer=self.save_optimizer),
            save_path / "snake.pt",
        )
        torch.save(
            bait_agent.get_full_state(include_optimizer=self.save_optimizer),
            save_path / "bait.pt",
        )
        with open(save_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Also update the manifest so load_latest() finds it after a crash
        self._manifest["latest"] = self.EMERGENCY_SUBDIR
        self._manifest["latest_episode"] = episode
        self._manifest.setdefault("checkpoints", [])
        # Replace or insert emergency entry
        self._manifest["checkpoints"] = [
            c for c in self._manifest["checkpoints"]
            if c["subdir"] != self.EMERGENCY_SUBDIR
        ]
        self._manifest["checkpoints"].append(
            {"episode": episode, "subdir": self.EMERGENCY_SUBDIR,
             "timestamp": state["timestamp"], "emergency": True}
        )
        self._write_manifest()

        return save_path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent valid checkpoint.

        Walks through checkpoints from newest to oldest until one loads
        successfully. Returns None if no checkpoints exist.

        Returns:
            Dict with keys:
                "episode"           — episode to resume from (+ 1)
                "snake_state"       — raw state dict for DQNAgent.load_full_state()
                "bait_state"        — raw state dict for DQNAgent.load_full_state()
                "training_meta"     — dict from training_state.json
            or None if no checkpoint found.
        """
        checkpoints = self._manifest.get("checkpoints", [])
        if not checkpoints:
            return None

        # Try from newest to oldest
        for entry in reversed(checkpoints):
            result = self._try_load(entry["subdir"])
            if result is not None:
                return result
            logger.warning(
                "Checkpoint '%s' is corrupt or incomplete — trying previous.",
                entry["subdir"],
            )

        logger.error("All checkpoints are corrupt. Starting fresh.")
        return None

    def list_checkpoints(self) -> list:
        """Return the manifest list of all known checkpoints."""
        return self._manifest.get("checkpoints", [])

    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _try_load(self, subdir: str) -> Optional[Dict[str, Any]]:
        """Attempt to load a checkpoint by subdirectory name. Returns None on failure."""
        save_path = self.checkpoint_dir / subdir
        snake_path = save_path / "snake.pt"
        bait_path = save_path / "bait.pt"
        meta_path = save_path / "training_state.json"

        for p in (snake_path, bait_path, meta_path):
            if not p.exists():
                return None

        try:
            snake_state = torch.load(snake_path, map_location="cpu", weights_only=False)
            bait_state = torch.load(bait_path, map_location="cpu", weights_only=False)
            with open(meta_path) as f:
                training_meta = json.load(f)
            return {
                "episode": training_meta["episode"],
                "snake_state": snake_state,
                "bait_state": bait_state,
                "training_meta": training_meta,
            }
        except Exception as exc:
            logger.warning("Failed to load checkpoint '%s': %s", subdir, exc)
            return None

    def _save_tensor(self, path: Path, data: Any) -> None:
        """Save a tensor/dict using atomic write if enabled."""
        if self.atomic_write:
            # Write to a sibling .tmp file, then atomically rename
            tmp_path = path.with_suffix(".pt.tmp")
            torch.save(data, tmp_path)
            os.replace(tmp_path, path)
        else:
            torch.save(data, path)

    def _save_json(self, path: Path, data: Dict) -> None:
        """Save a JSON dict using atomic write if enabled."""
        if self.atomic_write:
            tmp_path = path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest.json or return an empty manifest."""
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("manifest.json is corrupt — reinitialising.")
        return {"checkpoints": [], "latest": None, "latest_episode": 0}

    def _write_manifest(self) -> None:
        """Persist the current manifest to disk (atomic)."""
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        tmp_path = self.checkpoint_dir / (self.MANIFEST_FILE + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)

    def _prune_old(self) -> None:
        """Remove checkpoints beyond the keep_last_n rolling window."""
        # Never prune emergency checkpoints
        regular = [
            c for c in self._manifest["checkpoints"]
            if c.get("subdir") != self.EMERGENCY_SUBDIR
        ]
        # Sort by episode ascending; prune the oldest
        regular.sort(key=lambda c: c["episode"])
        to_remove = regular[: max(0, len(regular) - self.keep_last_n)]

        for entry in to_remove:
            stale_path = self.checkpoint_dir / entry["subdir"]
            if stale_path.exists():
                shutil.rmtree(stale_path, ignore_errors=True)
                logger.debug("Pruned old checkpoint: %s", stale_path)

        # Update manifest to reflect pruning
        removed_subdirs = {e["subdir"] for e in to_remove}
        self._manifest["checkpoints"] = [
            c for c in self._manifest["checkpoints"]
            if c["subdir"] not in removed_subdirs
        ]
        self._write_manifest()

    def _sync_to_drive(self, src_path: Path, subdir: str) -> None:
        """Mirror a checkpoint directory to the configured Google Drive path."""
        if not self.drive_sync_dir:
            return
        dst = self.drive_sync_dir / subdir
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_path, dst)
            # Also mirror the manifest
            shutil.copy(
                self.checkpoint_dir / self.MANIFEST_FILE,
                self.drive_sync_dir / self.MANIFEST_FILE,
            )
            logger.debug("Drive sync: %s → %s", src_path, dst)
        except Exception as exc:
            # Drive sync is best-effort; never crash training because of it
            logger.warning("Drive sync failed (non-fatal): %s", exc)
