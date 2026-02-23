#!/usr/bin/env python3
import argparse
import dataclasses
import glob
import logging
import logging.handlers
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple


# --------------------------------------------------------------------------------------
# Defaults (centralized)
# --------------------------------------------------------------------------------------

DEFAULT_INPUT = Path("/workspace/input")
DEFAULT_OUTPUT = Path("/workspace/output")
DEFAULT_PROGRESS_FILE = Path("/workspace/result.txt")

DEFAULT_WEIGHT_PATH_CANDIDATES = (
    Path("/runpod-volume/models/HYPIR_sd2.pth"),
    Path("/workspace/models/HYPIR_sd2.pth"),
    Path("/runpod-volume/HYPIR_sd2.pth"),
    Path("/workspace/HYPIR_sd2.pth"),
    Path("/workspace/HYPIR/HYPIR_sd2.pth"),
    Path.cwd() / "models" / "HYPIR_sd2.pth",
    Path.cwd() / "HYPIR_sd2.pth",
)

DEFAULT_BASE_MODEL_LOCAL_PATH_CANDIDATES = (
    Path("/runpod-volume/models/sd2-base"),
    Path("/workspace/models/sd2-base"),
    Path("/runpod-volume/models/stabilityai/stable-diffusion-2-1-base"),
    Path("/workspace/models/stabilityai/stable-diffusion-2-1-base"),
    Path("/workspace/HYPIR/models/stabilityai/stable-diffusion-2-1-base"),
    Path.cwd() / "HYPIR" / "models" / "stabilityai" / "stable-diffusion-2-1-base",
)

DEFAULT_HYPIR_REPO_PATH_CANDIDATES = (
    Path("/runpod-volume/HYPIR"),
    Path("/workspace/HYPIR"),
    Path.cwd() / "HYPIR",
)

DEFAULT_BASE_MODEL_HF_ID = "stabilityai/stable-diffusion-2-1-base"

DEFAULT_MODEL_T = 200
DEFAULT_COEFF_T = 200
DEFAULT_LORA_RANK = 256
DEFAULT_LORA_MODULES = ",".join(
    [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "conv_out",
        "proj_in",
        "proj_out",
        "ff.net.2",
        "ff.net.0.proj",
    ]
)

DEFAULT_SCALE_BY = "longest_side"
DEFAULT_UPSCALE = 4
DEFAULT_TARGET_LONGEST_SIDE = 4200

DEFAULT_PATCH_SIZE = 640
DEFAULT_STRIDE = 320

DEFAULT_PROMPT = ""
DEFAULT_SEED = 231

DEFAULT_LOG_DIR = Path("logs")

DEFAULT_IMAGE_EXTENSIONS = "png,jpg,jpeg,webp,bmp,tif,tiff"

DEFAULT_PROGRESS_CONSOLE_INTERVAL_S = 1.0

DEFAULT_PNG_COMPRESS_LEVEL = 3

DEFAULT_TASK_QUEUE_MAXSIZE = 0  # 0 means "infinite" for multiprocessing.Queue

DEFAULT_TORCH_NUM_THREADS = 0  # 0 = auto: max(1, C//G), clamped to <= 16
DEFAULT_TORCH_NUM_INTEROP_THREADS = 1
DEFAULT_MAX_TORCH_NUM_THREADS = 16

DEFAULT_MP_START_METHOD = "auto"  # auto|spawn|forkserver|fork
DEFAULT_LOG_LEVEL = "INFO"

# If you use compileall in the image, 'off' is usually best.
DEFAULT_PYCACHE_PREFIX = None  # set by presets


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve_path(value: str | Path) -> Path:
    """Resolve a path string or Path to an absolute Path.

    :param value: The path value to resolve.
    :return: Resolved absolute path.
    """
    return Path(value).expanduser().resolve()


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    """Clamp an integer value into a closed interval.

    :param value: Input value.
    :param min_value: Minimum allowed value.
    :param max_value: Maximum allowed value.
    :return: Clamped value.
    """
    return max(min_value, min(int(value), max_value))


def _auto_threads_per_worker(cpu_count: int, num_workers: int, max_threads: int) -> int:
    """Compute threads per worker using max(1, C // G), clamped to a maximum.

    :param cpu_count: Number of CPUs reported by the OS.
    :param num_workers: Number of worker processes.
    :param max_threads: Maximum allowed threads per worker.
    :return: Threads per worker.
    """
    c = int(cpu_count) if int(cpu_count) > 0 else 4
    g = max(1, int(num_workers))
    threads = max(1, c // g)
    return _clamp_int(threads, 1, int(max_threads))


def _write_atomic_text(path: Path, text: str) -> None:
    """Write a small text file atomically using os.replace.

    :param path: Target file path.
    :param text: File contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}_{int(time.time() * 1000)}")
    tmp.write_text(str(text), encoding="utf-8")
    os.replace(tmp, path)


def _first_existing_path(candidates: tuple[Path, ...]) -> Optional[Path]:
    """Return the first existing path from candidates.

    :param candidates: Candidate paths to check.
    :return: First existing path or None.
    """
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _default_weight_path() -> Path:
    """Determine the default weight file path.

    :return: Existing weight path.
    """
    found_non_files: list[Path] = []
    for p in DEFAULT_WEIGHT_PATH_CANDIDATES:
        if not p.exists():
            continue
        rp = p.resolve()
        if rp.is_file():
            return rp
        found_non_files.append(rp)

    err = (
        "No HYPIR weight file found. Checked:\n"
        + "\n".join(f"- {p}" for p in DEFAULT_WEIGHT_PATH_CANDIDATES)
    )
    if found_non_files:
        err += "\nExisting non-file paths ignored:\n" + "\n".join(f"- {p}" for p in found_non_files)
    raise SystemExit(err)


def _default_base_model_path() -> str:
    """Determine the default base model path or HF id.

    Resolution order:
    1) Env var HYPIR_BASE_MODEL_PATH
    2) Local base model candidates
    3) HF model id default

    :return: Local path as string or HF model id.
    """
    if os.environ.get("HYPIR_BASE_MODEL_PATH"):
        return os.environ["HYPIR_BASE_MODEL_PATH"]

    found = _first_existing_path(DEFAULT_BASE_MODEL_LOCAL_PATH_CANDIDATES)
    if found:
        return str(found)

    return DEFAULT_BASE_MODEL_HF_ID


def _apply_hf_environment(hf_token: Optional[str], local_files_only: bool) -> None:
    """Configure Hugging Face environment variables.

    :param hf_token: Optional token for gated models.
    :param local_files_only: If True, enforce offline mode.
    """
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _apply_pycache_prefix(pycache_prefix: str, logger: logging.Logger) -> None:
    """Configure Python bytecode cache prefix.

    If you built your image with compileall and want to use __pycache__ shipped in the image,
    set this to 'off'.

    :param pycache_prefix: 'auto', 'off', or an explicit directory path.
    :param logger: Logger instance.
    """
    raw = (pycache_prefix or "").strip()
    if not raw:
        raw = "off"

    if raw.lower() in {"off", "false", "0", "disable", "disabled"}:
        logger.info("PYTHONPYCACHEPREFIX disabled.")
        return

    if "PYTHONPYCACHEPREFIX" in os.environ and raw.lower() == "auto":
        logger.info("PYTHONPYCACHEPREFIX already set: %s", os.environ["PYTHONPYCACHEPREFIX"])
        return

    def _try_set(prefix_dir: Path, reason: str) -> bool:
        try:
            prefix_dir.mkdir(parents=True, exist_ok=True)
            test_file = prefix_dir / f".write_test_{os.getpid()}"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            os.environ["PYTHONPYCACHEPREFIX"] = str(prefix_dir)
            logger.info("PYTHONPYCACHEPREFIX set to %s (%s)", prefix_dir, reason)
            return True
        except Exception as exc:
            logger.debug("Failed to set PYTHONPYCACHEPREFIX to %s: %s", prefix_dir, exc)
            return False

    if raw.lower() == "auto":
        candidates = [
            Path("/dev/shm/pycache"),
            Path("/tmp/pycache"),
        ]
        for c in candidates:
            if _try_set(c, reason="auto"):
                return
        logger.info("PYTHONPYCACHEPREFIX not set (no writable candidate found).")
        return

    explicit = Path(os.path.expanduser(raw)).resolve()
    if not _try_set(explicit, reason="explicit"):
        logger.warning("Could not use pycache prefix: %s (leaving default)", explicit)


def _ensure_hypir_repo_on_path(hypir_repo_path: str | Path | None = None) -> Path:
    """Ensure the HYPIR repo is available and added to sys.path.

    Resolution order:
    1) CLI arg hypir_repo_path (if provided)
    2) Env var HYPIR_REPO_PATH
    3) Default candidates list

    :param hypir_repo_path: Optional repo path from CLI.
    :return: Resolved repo path.
    """
    if hypir_repo_path:
        candidate = _resolve_path(hypir_repo_path)
        if not candidate.exists() or not candidate.is_dir():
            raise SystemExit(f"HYPIR repo not found at {candidate}")
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return candidate

    env_value = os.environ.get("HYPIR_REPO_PATH")
    if env_value:
        candidate = _resolve_path(env_value)
        if not candidate.exists() or not candidate.is_dir():
            raise SystemExit(f"HYPIR repo not found at {candidate} (from HYPIR_REPO_PATH)")
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return candidate

    found = _first_existing_path(DEFAULT_HYPIR_REPO_PATH_CANDIDATES)
    if not found:
        raise SystemExit(
            "HYPIR repo not found. Checked:\n" + "\n".join(f"- {p}" for p in DEFAULT_HYPIR_REPO_PATH_CANDIDATES)
        )
    if not found.is_dir():
        raise SystemExit(f"HYPIR repo path is not a directory: {found}")

    if str(found) not in sys.path:
        sys.path.insert(0, str(found))
    return found


def _print_hf_hint_if_needed(exc: Exception, logger: Optional[logging.Logger] = None) -> None:
    """Print a helpful hint for common Hugging Face gated-model access errors.

    :param exc: The exception thrown while loading models.
    :param logger: Optional logger for structured output.
    """
    msg = str(exc)
    if "401" in msg or "gated" in msg or "Repository Not Found" in msg:
        text = (
            "Hugging Face access error. The base model might be gated and requires login.\n"
            "Fix options:\n"
            "1) Run `huggingface-cli login` and accept the SD2 license on HF.\n"
            "2) Or pass `--hf_token YOUR_TOKEN`.\n"
            "3) Or download the model and pass `--base_model_path /path/to/sd2`.\n"
        )
        if logger:
            logger.error(text)
        else:
            print("\n" + text)


def _is_existing_dir(path_or_id: str) -> bool:
    """Check whether the given string is an existing directory path.

    :param path_or_id: Path or HF repo id.
    :return: True if it is an existing directory.
    """
    try:
        p = Path(path_or_id).expanduser()
        return p.exists() and p.is_dir()
    except Exception:
        return False


def _looks_like_hf_repo_id(path_or_id: str) -> bool:
    """Heuristic check whether a string looks like a Hugging Face repo id.

    :param path_or_id: Path or HF repo id.
    :return: True if it looks like a HF repo id.
    """
    if "://" in path_or_id:
        return False
    if path_or_id.count("/") != 1:
        return False
    a, b = path_or_id.split("/", 1)
    if not a or not b:
        return False
    if any(ch.isspace() for ch in path_or_id):
        return False
    return True


def _maybe_snapshot_download_base_model(
    base_model_path: str,
    local_files_only: bool,
    logger: logging.Logger,
) -> str:
    """Resolve base model into a local directory if it is a HF repo id.

    If base_model_path is already an existing directory, it is returned unchanged.
    If it looks like a HF repo id, snapshot_download is used to obtain a local folder.

    :param base_model_path: Local path or HF repo id.
    :param local_files_only: If True, do not download (only use cache).
    :param logger: Logger instance.
    :return: Local directory path as string.
    """
    if _is_existing_dir(base_model_path):
        return str(_resolve_path(base_model_path))

    if not _looks_like_hf_repo_id(base_model_path):
        return base_model_path

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        logger.warning(
            "huggingface_hub not available, cannot snapshot_download. Passing HF id directly. Error: %s",
            exc,
        )
        return base_model_path

    logger.info("Resolving base model via snapshot_download: %s (local_files_only=%s)", base_model_path, local_files_only)
    try:
        local_dir = snapshot_download(
            repo_id=base_model_path,
            local_files_only=local_files_only,
        )
        return str(_resolve_path(local_dir))
    except Exception as exc:
        _print_hf_hint_if_needed(exc, logger=logger)
        raise


def _iter_files_for_warmup(root: Path) -> Iterable[Path]:
    """Iterate files under a directory for warmup.

    :param root: Directory path.
    :return: Iterator over file paths.
    """
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _warmup_read_file(path: Path, block_size: int = 16 * 1024 * 1024) -> int:
    """Read a file sequentially to warm OS page cache.

    :param path: File to read.
    :param block_size: Read block size in bytes.
    :return: Total bytes read.
    """
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            total += len(chunk)
    return total


def _warmup_paths(paths: list[Path], io_workers: int, logger: logging.Logger) -> Tuple[int, float]:
    """Warm OS page cache for a list of files by reading them concurrently.

    :param paths: List of file paths.
    :param io_workers: Number of worker threads.
    :param logger: Logger instance.
    :return: (total_bytes_read, elapsed_seconds)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        ps = str(p)
        if ps not in seen and p.exists() and p.is_file():
            seen.add(ps)
            unique_paths.append(p)

    if not unique_paths:
        return 0, 0.0

    start = time.perf_counter()
    total_bytes = 0

    logger.info("Warmup: reading %d files into OS page cache with %d threads ...", len(unique_paths), io_workers)
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
        futures = [ex.submit(_warmup_read_file, p) for p in unique_paths]
        for fut in as_completed(futures):
            total_bytes += int(fut.result())

    elapsed = time.perf_counter() - start
    mb = total_bytes / (1024 * 1024) if total_bytes > 0 else 0.0
    rate = (mb / elapsed) if elapsed > 0 else 0.0
    logger.info("Warmup done: %.1f MiB read in %.2fs (%.1f MiB/s)", mb, elapsed, rate)
    return total_bytes, elapsed


def _human_bytes(num: int) -> str:
    """Format bytes as a human readable string.

    :param num: Bytes.
    :return: String.
    """
    n = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} B"


def _stage_file_to_fast_storage(src: Path, mode: str, logger: logging.Logger) -> Path:
    """Stage a file into fast local storage to avoid slow network volumes.

    Locations (in order):
    1) /dev/shm (RAM disk) if enough space
    2) /tmp (local container disk) if enough space

    Destination name is deterministic based on (size, mtime_ns) to allow reuse within the same pod.

    :param src: Source file path.
    :param mode: 'auto', 'on', or 'off'.
    :param logger: Logger instance.
    :return: Path to staged file, or original src if not staged.
    """
    mode = (mode or "auto").lower().strip()
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Invalid staging mode: {mode}")

    if mode == "off":
        return src

    stat = src.stat()
    tag = f"{stat.st_size}_{stat.st_mtime_ns}"
    candidates = [Path("/dev/shm"), Path("/tmp")]

    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue
        try:
            usage = shutil.disk_usage(str(root))
        except Exception:
            continue

        if stat.st_size > usage.free:
            continue

        dst = root / f"{src.stem}_staged_{tag}{src.suffix}"
        if dst.exists() and dst.is_file() and dst.stat().st_size == stat.st_size:
            logger.info("Staged file already present: %s", dst)
            return dst.resolve()

        logger.info("Staging file to fast storage: %s -> %s (%s)", src, dst, _human_bytes(stat.st_size))
        tmp = dst.with_name(f".{dst.name}.tmp_{os.getpid()}_{int(time.time() * 1000)}")
        t0 = time.perf_counter()
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

        logger.info("Staging done in %.2fs (%s)", time.perf_counter() - t0, dst)
        return dst.resolve()

    if mode == "on":
        logger.warning("Staging requested but no suitable fast storage location had enough space. Using original file.")
    return src


def _parse_csv_ints(value: Optional[str]) -> Optional[list[int]]:
    """Parse a comma-separated list of ints.

    :param value: CSV string.
    :return: List of ints or None.
    """
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def _parse_csv_strs(value: str) -> list[str]:
    """Parse a comma-separated list of strings.

    :param value: CSV string.
    :return: List of strings.
    """
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _collect_images_from_dir(root: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    """Collect image files from a directory.

    :param root: Directory root.
    :param recursive: If True, search recursively.
    :param extensions: Allowed extensions (lowercase, without dot).
    :return: List of image paths.
    """
    it = root.rglob("*") if recursive else root.glob("*")
    out: list[Path] = []
    for p in it:
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext in extensions:
            out.append(p.resolve())
    return out


def _collect_images_from_glob(pattern: str, extensions: set[str]) -> list[Path]:
    """Collect image files from a glob pattern (supports absolute patterns).

    :param pattern: Glob pattern.
    :param extensions: Allowed extensions.
    :return: List of image paths.
    """
    pat = os.path.expanduser(pattern)
    matched = glob.glob(pat, recursive=("**" in pat))
    out: list[Path] = []
    for m in sorted(matched):
        p = Path(m).resolve()
        if p.is_file() and p.suffix.lower().lstrip(".") in extensions:
            out.append(p)
    return out


def _collect_input_images(
    inputs: list[str],
    recursive: bool,
    extensions_csv: str,
    logger: logging.Logger,
) -> Tuple[list[Path], Optional[Path]]:
    """Collect input images from files, directories, and/or glob patterns.

    :param inputs: Input arguments.
    :param recursive: Recursive directory scan.
    :param extensions_csv: Allowed extensions as CSV.
    :param logger: Logger instance.
    :return: (list_of_images, common_root_if_any)
    """
    extensions = {e.strip().lower().lstrip(".") for e in extensions_csv.split(",") if e.strip()}
    if not extensions:
        raise SystemExit("No valid extensions provided.")

    all_images: list[Path] = []

    for raw in inputs:
        val = os.path.expanduser(raw)
        p = Path(val)

        if p.exists() and p.is_file():
            ext = p.suffix.lower().lstrip(".")
            if ext not in extensions:
                logger.warning("Skipping file with disallowed extension: %s", p)
                continue
            all_images.append(p.resolve())
            continue

        if p.exists() and p.is_dir():
            all_images.extend(_collect_images_from_dir(p.resolve(), recursive=recursive, extensions=extensions))
            continue

        if any(ch in val for ch in ["*", "?", "["]) or "**" in val:
            all_images.extend(_collect_images_from_glob(val, extensions=extensions))
            continue

        logger.warning("Input not found (file/dir/glob): %s", raw)

    seen: set[str] = set()
    uniq: list[Path] = []
    for p in all_images:
        ps = str(p)
        if ps not in seen:
            seen.add(ps)
            uniq.append(p)

    uniq.sort()
    if not uniq:
        raise SystemExit("No input images found.")

    common_root: Optional[Path] = None
    try:
        common_root = Path(os.path.commonpath([str(p.parent) for p in uniq])).resolve()
    except Exception:
        common_root = None

    return uniq, common_root


def _build_output_path(input_path: Path, output_dir: Path, keep_relative_to: Optional[Path]) -> Path:
    """Build output path for an input image.

    :param input_path: Input image path.
    :param output_dir: Output directory root.
    :param keep_relative_to: If set, preserve directory structure relative to this root.
    :return: Output image path.
    """
    if keep_relative_to:
        try:
            rel = input_path.relative_to(keep_relative_to)
            rel_parent = rel.parent
        except Exception:
            rel_parent = Path()
    else:
        rel_parent = Path()

    out_dir = (output_dir / rel_parent).resolve()
    out_name = f"{input_path.stem}_hypir.png"
    return (out_dir / out_name).resolve()


def _save_png(pil_image, output_path: str | Path, compress_level: int) -> None:
    """Save a PIL image as PNG with a specific compression level.

    :param pil_image: PIL Image instance.
    :param output_path: Output file path.
    :param compress_level: PNG compress_level (0-9).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_p, format="PNG", compress_level=int(compress_level), optimize=False)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Check whether an exception looks like a CUDA out-of-memory error.

    :param exc: Exception instance.
    :return: True if it looks like CUDA OOM.
    """
    msg = str(exc).lower()
    return ("cuda out of memory" in msg) or ("out of memory" in msg and "cuda" in msg)


def _patch_candidates_for_vram(vram_bytes: int, base_patch: int) -> list[int]:
    """Return patch size candidates based on GPU VRAM.

    :param vram_bytes: Total VRAM bytes.
    :param base_patch: Fallback patch size.
    :return: List of patch sizes to try (largest first).
    """
    gib = 1024 * 1024 * 1024
    if vram_bytes >= 22 * gib:
        candidates = [1024, 896, 768, base_patch]
    elif vram_bytes >= 16 * gib:
        candidates = [896, 768, base_patch]
    elif vram_bytes >= 12 * gib:
        candidates = [768, base_patch]
    else:
        candidates = [base_patch]

    out: list[int] = []
    seen: set[int] = set()
    for p in candidates:
        if int(p) not in seen:
            out.append(int(p))
            seen.add(int(p))
    return out


# --------------------------------------------------------------------------------------
# Profiling primitives
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class RunningStats:
    """Accumulate basic statistics for a stream of float values."""

    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, value: float) -> None:
        """Update statistics with a new value.

        :param value: New value to include.
        """
        self.count += 1
        self.total += float(value)
        if value < self.min_value:
            self.min_value = float(value)
        if value > self.max_value:
            self.max_value = float(value)

    def mean(self) -> float:
        """Compute mean value.

        :return: Mean.
        """
        if self.count <= 0:
            return 0.0
        return self.total / self.count

    def as_dict(self) -> dict:
        """Serialize to dict.

        :return: Dict with count/total/min/max/mean.
        """
        return {
            "count": self.count,
            "total_s": self.total,
            "min_s": (0.0 if self.min_value == float("inf") else self.min_value),
            "max_s": (0.0 if self.max_value == float("-inf") else self.max_value),
            "mean_s": self.mean(),
        }


@dataclasses.dataclass
class TopKSlow:
    """Keep the top-k slowest items."""

    k: int = 10
    items: list[tuple[float, str]] = dataclasses.field(default_factory=list)

    def update(self, value: float, label: str) -> None:
        """Insert a candidate in the top-k list.

        :param value: Time value in seconds.
        :param label: Label (e.g. file name).
        """
        self.items.append((float(value), str(label)))
        self.items.sort(key=lambda x: x[0], reverse=True)
        if len(self.items) > self.k:
            self.items = self.items[: self.k]

    def as_list(self) -> list[dict]:
        """Serialize to list.

        :return: List of dicts.
        """
        return [{"seconds": v, "label": lbl} for v, lbl in self.items]


class StageTimer:
    """Context manager for timing a stage with perf_counter."""

    def __init__(self, stats: RunningStats) -> None:
        """Create a stage timer.

        :param stats: RunningStats instance to update.
        """
        self._stats = stats
        self._t0: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.perf_counter() - self._t0
        self._stats.update(dt)


class CudaEventTimer:
    """Measure CUDA kernel time using CUDA events (best-effort)."""

    def __init__(self, enabled: bool, gpu_index: int) -> None:
        """Create a CUDA event timer.

        :param enabled: Whether CUDA timing should be enabled.
        :param gpu_index: GPU index for context.
        """
        self._enabled = bool(enabled)
        self._gpu_index = int(gpu_index)
        self._start = None
        self._end = None
        self._elapsed_ms: Optional[float] = None

    def __enter__(self) -> "CudaEventTimer":
        if not self._enabled:
            return self
        import torch

        with torch.cuda.device(self._gpu_index):
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled or self._start is None or self._end is None:
            return
        import torch

        with torch.cuda.device(self._gpu_index):
            self._end.record()
            self._end.synchronize()
            self._elapsed_ms = float(self._start.elapsed_time(self._end))

    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds (CUDA time).

        :return: Elapsed seconds (0.0 if not available).
        """
        if self._elapsed_ms is None:
            return 0.0
        return self._elapsed_ms / 1000.0


# --------------------------------------------------------------------------------------
# Worker config
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WorkerConfig:
    """Serializable worker configuration."""

    hypir_repo_path: str
    run_id: str
    base_model_path: str
    weight_path: str
    lora_modules: list[str]
    lora_rank: int
    model_t: int
    coeff_t: int
    prompt: str
    scale_by: str
    upscale: int
    target_longest_side: Optional[int]
    patch_size: int
    stride: int
    auto_patch: bool
    seed: int
    device_prefix: str
    prefetch: int
    save_async: bool
    png_compress_level: int
    save_queue_size: int
    torch_num_threads: int
    torch_num_interop_threads: int
    accurate_cuda_timing: bool
    cudnn_benchmark: bool
    tf32: bool


# --------------------------------------------------------------------------------------
# Progress reporting (main process)
# --------------------------------------------------------------------------------------


def _progress_writer(
    progress_queue,
    stop_event: threading.Event,
    progress_path: Path,
    total: int,
    console_interval_s: float,
) -> None:
    """Write a progress file as 'X/Y' and optionally print it to stdout.

    :param progress_queue: Multiprocessing queue receiving integer increments or None sentinel.
    :param stop_event: Stop event to request termination.
    :param progress_path: Path to progress file.
    :param total: Total number of tasks/images.
    :param console_interval_s: Minimum seconds between console prints. 0 prints every update; negative disables.
    """
    done = 0
    total_i = max(0, int(total))
    last_print_t = 0.0

    _write_atomic_text(progress_path, f"{done}/{total_i}")

    while not stop_event.is_set():
        try:
            msg = progress_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if msg is None:
            break

        try:
            inc = int(msg)
        except Exception:
            inc = 0

        if inc <= 0:
            continue

        done += inc
        if done > total_i:
            done = total_i

        text = f"{done}/{total_i}"
        _write_atomic_text(progress_path, text)

        if console_interval_s >= 0:
            now = time.perf_counter()
            if console_interval_s == 0 or (now - last_print_t) >= console_interval_s or done >= total_i:
                print(text, flush=True)
                last_print_t = now

        if done >= total_i:
            break

    _write_atomic_text(progress_path, f"{done}/{total_i}")


# --------------------------------------------------------------------------------------
# Worker implementation
# --------------------------------------------------------------------------------------


class ImagePrefetcher(threading.Thread):
    """Prefetch images from a multiprocessing task queue into an in-process queue."""

    def __init__(
        self,
        gpu_index: int,
        tasks_queue,
        loaded_queue: "queue.Queue[Optional[tuple[str, str, object, float, Optional[str]]]]",
        logger: logging.Logger,
    ) -> None:
        """Initialize the prefetcher.

        :param gpu_index: GPU index for logging context.
        :param tasks_queue: Multiprocessing queue producing (input_path, output_path) or None sentinel.
        :param loaded_queue: Thread queue receiving (input_path, output_path, lq_tensor, load_s, error) or None sentinel.
        :param logger: Logger instance.
        """
        super().__init__(daemon=True)
        self._gpu_index = int(gpu_index)
        self._tasks_queue = tasks_queue
        self._loaded_queue = loaded_queue
        self._logger = logger
        self._load_stats = RunningStats()
        self._decode_failures = 0

    def run(self) -> None:
        """Run the prefetcher loop."""
        from PIL import Image
        import torch

        while True:
            task = self._tasks_queue.get()
            if task is None:
                self._loaded_queue.put(None)
                return

            input_path, output_path = task
            t0 = time.perf_counter()
            try:
                img = Image.open(input_path).convert("RGB")

                w, h = img.size
                buf = img.tobytes()

                try:
                    storage = torch.UntypedStorage.from_buffer(buf, dtype=torch.uint8)
                    x = torch.ByteTensor(storage)
                except Exception:
                    storage = torch.ByteStorage.from_buffer(buf)
                    x = torch.ByteTensor(storage)

                x = x.view(h, w, 3).permute(2, 0, 1).contiguous()
                lq_tensor = x.float().div(255.0).unsqueeze(0)

                load_s = time.perf_counter() - t0
                self._load_stats.update(load_s)
                self._loaded_queue.put((input_path, output_path, lq_tensor, load_s, None))
            except Exception as exc:
                self._decode_failures += 1
                load_s = time.perf_counter() - t0
                self._load_stats.update(load_s)
                self._logger.error("[gpu=%d] Failed to decode image: %s (%s)", self._gpu_index, input_path, exc)
                self._loaded_queue.put((input_path, output_path, None, load_s, str(exc)))

    def stats(self) -> dict:
        """Return prefetcher statistics.

        :return: Dict with load stats and failures.
        """
        return {
            "image_load": self._load_stats.as_dict(),
            "decode_failures": self._decode_failures,
        }


class AsyncSaver(threading.Thread):
    """Asynchronous image saver."""

    def __init__(
        self,
        gpu_index: int,
        save_queue: "queue.Queue[Optional[tuple[object, str, str]]]",
        logger: logging.Logger,
        png_compress_level: int,
        progress_queue,
    ) -> None:
        """Initialize the saver.

        :param gpu_index: GPU index for logging.
        :param save_queue: Queue receiving (pil_image, output_path, input_path) or None sentinel.
        :param logger: Logger instance.
        :param png_compress_level: PNG compression level (0-9).
        :param progress_queue: Multiprocessing queue receiving increments for completed tasks.
        """
        super().__init__(daemon=True)
        self._gpu_index = int(gpu_index)
        self._save_queue = save_queue
        self._logger = logger
        self._save_stats = RunningStats()
        self._save_failures = 0
        self._png_compress_level = _clamp_int(int(png_compress_level), 0, 9)
        self._progress_queue = progress_queue

    def run(self) -> None:
        """Run save loop."""
        while True:
            item = self._save_queue.get()
            if item is None:
                return

            pil_image, output_path, input_path = item
            t0 = time.perf_counter()
            try:
                _save_png(pil_image, output_path, compress_level=self._png_compress_level)
                dt = time.perf_counter() - t0
                self._save_stats.update(dt)
            except Exception as exc:
                self._save_failures += 1
                self._logger.error(
                    "[gpu=%d] Failed to save output: %s (from %s) (%s)",
                    self._gpu_index,
                    output_path,
                    input_path,
                    exc,
                )
            finally:
                try:
                    self._progress_queue.put(1)
                except Exception:
                    pass

    def stats(self) -> dict:
        """Return saver statistics.

        :return: Dict with save stats and failures.
        """
        return {
            "save": self._save_stats.as_dict(),
            "save_failures": self._save_failures,
        }


def _configure_worker_logging(log_queue, level: str) -> logging.Logger:
    """Configure worker process logging to forward records to the main process.

    :param log_queue: Multiprocessing queue for logs.
    :param level: Log level string.
    :return: Logger instance.
    """
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    qh = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(qh)
    logger.propagate = False
    return logger


def _worker_main(
    gpu_index: int,
    cfg: WorkerConfig,
    tasks_queue,
    results_queue,
    log_queue,
    progress_queue,
    log_level: str,
) -> None:
    """Worker process entry point.

    :param gpu_index: GPU index used by this worker.
    :param cfg: Worker configuration.
    :param tasks_queue: Multiprocessing queue with tasks (input_path, output_path) or None sentinel.
    :param results_queue: Multiprocessing queue for returning worker summary stats.
    :param log_queue: Multiprocessing queue for logging.
    :param progress_queue: Multiprocessing queue for progress increments.
    :param log_level: Log level string.
    """
    # Prevent hidden CPU oversubscription (important on multi-GPU nodes).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logger = _configure_worker_logging(log_queue, level=log_level)

    try:
        _ensure_hypir_repo_on_path(cfg.hypir_repo_path)

        import torch

        if cfg.cudnn_benchmark:
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        if cfg.tf32 and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
        try:
            torch.set_num_interop_threads(max(1, int(cfg.torch_num_interop_threads)))
        except Exception:
            pass

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        device = f"{cfg.device_prefix}:{gpu_index}" if cfg.device_prefix == "cuda" else cfg.device_prefix
        logger.info(
            "[gpu=%d] Worker starting device=%s threads=%d interop=%d prefetch=%d save_async=%s png_c=%d weight=%s",
            gpu_index,
            device,
            int(cfg.torch_num_threads),
            int(cfg.torch_num_interop_threads),
            int(cfg.prefetch),
            bool(cfg.save_async),
            int(cfg.png_compress_level),
            str(cfg.weight_path),
        )

        # Import + init model first (avoid competing with image I/O during init).
        t_import0 = time.perf_counter()
        from HYPIR.enhancer.sd2 import SD2Enhancer
        logger.info("[gpu=%d] import SD2Enhancer took %.2fs", gpu_index, time.perf_counter() - t_import0)

        model_init_stats = RunningStats()
        with StageTimer(model_init_stats):
            model = SD2Enhancer(
                base_model_path=cfg.base_model_path,
                weight_path=cfg.weight_path,
                lora_modules=cfg.lora_modules,
                lora_rank=cfg.lora_rank,
                model_t=cfg.model_t,
                coeff_t=cfg.coeff_t,
                device=device,
            )
            logger.info("[gpu=%d] init_models() starting ...", gpu_index)
            try:
                model.init_models()
            except Exception as exc:
                _print_hf_hint_if_needed(exc, logger=logger)
                raise
            logger.info("[gpu=%d] init_models() done.", gpu_index)

        # Tiling policy:
        # - Base overlap derived from user patch/stride (keeps seam behavior stable).
        # - In auto_patch mode, try larger patches (4090 benefits), keep overlap constant.
        base_patch = int(cfg.patch_size)
        base_stride = int(cfg.stride)
        overlap = base_patch - base_stride
        if overlap <= 0:
            overlap = max(16, base_patch // 8)
        overlap = _clamp_int(overlap, 16, max(16, base_patch - 1))

        patch_candidates: list[int] = [base_patch]
        if cfg.auto_patch and cfg.device_prefix == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(gpu_index)
                patch_candidates = _patch_candidates_for_vram(int(props.total_memory), base_patch=base_patch)
            except Exception:
                patch_candidates = [base_patch]

        current_patch_idx = 0
        current_patch = int(patch_candidates[current_patch_idx])
        current_stride = int(current_patch - overlap) if cfg.auto_patch else base_stride
        current_stride = max(1, current_stride)

        logger.info(
            "[gpu=%d] tiling: auto_patch=%s base=(patch=%d stride=%d overlap=%d) candidates=%s initial=(patch=%d stride=%d)",
            gpu_index,
            bool(cfg.auto_patch),
            base_patch,
            base_stride,
            overlap,
            patch_candidates,
            current_patch,
            current_stride,
        )

        # Start prefetcher + saver after model init.
        loaded_queue: "queue.Queue[Optional[tuple[str, str, object, float, Optional[str]]]]" = queue.Queue(
            maxsize=max(1, int(cfg.prefetch))
        )
        save_queue: "queue.Queue[Optional[tuple[object, str, str]]]" = queue.Queue(maxsize=max(1, int(cfg.save_queue_size)))

        prefetcher = ImagePrefetcher(
            gpu_index=gpu_index,
            tasks_queue=tasks_queue,
            loaded_queue=loaded_queue,
            logger=logger,
        )
        saver: Optional[AsyncSaver] = None
        if cfg.save_async:
            saver = AsyncSaver(
                gpu_index=gpu_index,
                save_queue=save_queue,
                logger=logger,
                png_compress_level=int(cfg.png_compress_level),
                progress_queue=progress_queue,
            )

        prefetcher.start()
        if saver:
            saver.start()

        gpu_mem = {}
        if torch.cuda.is_available() and cfg.device_prefix == "cuda":
            try:
                with torch.cuda.device(gpu_index):
                    torch.cuda.synchronize()
                    gpu_mem = {
                        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
            except Exception:
                gpu_mem = {}

        wait_for_input_stats = RunningStats()
        enhance_wall_stats = RunningStats()
        enhance_cuda_stats = RunningStats()
        push_save_stats = RunningStats()
        total_image_stats = RunningStats()
        slowest_total = TopKSlow(k=10)
        processed_ok = 0
        failures = 0

        logger.info("[gpu=%d] Starting batch processing loop ...", gpu_index)
        while True:
            t_total0 = time.perf_counter()

            t_wait0 = time.perf_counter()
            item = loaded_queue.get()
            wait_s = time.perf_counter() - t_wait0
            wait_for_input_stats.update(wait_s)

            if item is None:
                break

            input_path, output_path, lq_tensor, load_s, decode_error = item
            wall_s = 0.0
            cuda_s = 0.0

            try:
                if decode_error is not None or lq_tensor is None:
                    raise RuntimeError(f"Decode failed: {decode_error}")

                while True:
                    try:
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            with CudaEventTimer(
                                enabled=(cfg.accurate_cuda_timing and torch.cuda.is_available() and cfg.device_prefix == "cuda"),
                                gpu_index=gpu_index,
                            ) as ct:
                                result_list = model.enhance(
                                    lq=lq_tensor,
                                    prompt=cfg.prompt,
                                    scale_by=cfg.scale_by,
                                    upscale=cfg.upscale,
                                    target_longest_side=cfg.target_longest_side,
                                    patch_size=int(current_patch),
                                    stride=int(current_stride),
                                    return_type="pil",
                                )
                            cuda_s = ct.elapsed_seconds()
                        wall_s = time.perf_counter() - t0
                        enhance_wall_stats.update(wall_s)
                        if cuda_s > 0.0:
                            enhance_cuda_stats.update(cuda_s)
                        break
                    except Exception as exc:
                        if (
                            cfg.auto_patch
                            and cfg.device_prefix == "cuda"
                            and torch.cuda.is_available()
                            and _is_cuda_oom(exc)
                            and (current_patch_idx + 1) < len(patch_candidates)
                        ):
                            current_patch_idx += 1
                            current_patch = int(patch_candidates[current_patch_idx])
                            current_stride = max(1, int(current_patch - overlap))
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            logger.warning(
                                "[gpu=%d] CUDA OOM on %s -> fallback tiling to patch=%d stride=%d",
                                gpu_index,
                                input_path,
                                int(current_patch),
                                int(current_stride),
                            )
                            continue
                        raise

                result = result_list[0]
                t_push0 = time.perf_counter()
                if saver:
                    save_queue.put((result, output_path, input_path))
                else:
                    _save_png(result, output_path, compress_level=int(cfg.png_compress_level))
                    try:
                        progress_queue.put(1)
                    except Exception:
                        pass
                push_s = time.perf_counter() - t_push0
                push_save_stats.update(push_s)

                processed_ok += 1
                total_s = time.perf_counter() - t_total0
                total_image_stats.update(total_s)
                slowest_total.update(total_s, label=str(input_path))

                if processed_ok % 10 == 0:
                    logger.info(
                        "[gpu=%d] ok=%d fail=%d last_total=%.3fs wait=%.3fs load=%.3fs enhance=%.3fs cuda=%.3fs patch=%d stride=%d",
                        gpu_index,
                        processed_ok,
                        failures,
                        total_s,
                        wait_s,
                        float(load_s),
                        float(wall_s),
                        float(cuda_s),
                        int(current_patch),
                        int(current_stride),
                    )

            except Exception as exc:
                failures += 1
                logger.error("[gpu=%d] Failed processing %s: %s", gpu_index, input_path, exc)
                try:
                    progress_queue.put(1)
                except Exception:
                    pass
                continue

        if saver:
            save_queue.put(None)
            saver.join(timeout=600)
        prefetcher.join(timeout=600)

        if torch.cuda.is_available() and cfg.device_prefix == "cuda":
            try:
                with torch.cuda.device(gpu_index):
                    torch.cuda.synchronize()
            except Exception:
                pass

        summary = {
            "gpu_index": int(gpu_index),
            "device": device,
            "processed": int(processed_ok),
            "failures": int(failures),
            "model_init": model_init_stats.as_dict(),
            "wait_for_input": wait_for_input_stats.as_dict(),
            "enhance_wall": enhance_wall_stats.as_dict(),
            "enhance_cuda": enhance_cuda_stats.as_dict(),
            "push_save": push_save_stats.as_dict(),
            "total_per_image": total_image_stats.as_dict(),
            "slowest_total": slowest_total.as_list(),
            "prefetcher": prefetcher.stats(),
            "saver": (
                saver.stats()
                if saver
                else {
                    "save": {"count": 0, "total_s": 0.0, "min_s": 0.0, "max_s": 0.0, "mean_s": 0.0},
                    "save_failures": 0,
                }
            ),
            "gpu_memory": gpu_mem,
            "tiling": {"patch": int(current_patch), "stride": int(current_stride), "overlap": int(overlap)},
        }
        results_queue.put(summary)
        logger.info("[gpu=%d] Worker finished. ok=%d failures=%d", gpu_index, processed_ok, failures)

    except Exception as exc:
        try:
            logger.exception("[gpu=%d] Worker crashed: %s", gpu_index, exc)
        except Exception:
            pass
        results_queue.put(
            {
                "gpu_index": int(gpu_index),
                "device": f"{cfg.device_prefix}:{gpu_index}" if cfg.device_prefix == "cuda" else cfg.device_prefix,
                "processed": 0,
                "failures": 1,
                "crash": str(exc),
            }
        )
        raise


# --------------------------------------------------------------------------------------
# Logging setup (main process)
# --------------------------------------------------------------------------------------


def _setup_main_logging(run_id: str, log_level: str) -> tuple[logging.Logger, logging.Handler, logging.Handler, Path]:
    """Setup main process logging handlers (file + console).

    :param run_id: Run identifier.
    :param log_level: Log level string.
    :return: (logger, file_handler, console_handler, log_file_path)
    """
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = (DEFAULT_LOG_DIR / f"run_{run_id}.log").resolve()

    level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(processName)s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    logger = logging.getLogger("hypir_batch")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = True

    return logger, file_handler, console_handler, log_file


def _configure_main_root_logger_to_queue(mp_log_queue, log_level: str) -> None:
    """Configure main root logger to write into the multiprocessing log queue.

    :param mp_log_queue: Multiprocessing queue.
    :param log_level: Log level string.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.addHandler(logging.handlers.QueueHandler(mp_log_queue))
    root.propagate = False


def _format_run_id(now: Optional[float] = None) -> str:
    """Create a run id string without the year.

    :param now: Optional epoch timestamp.
    :return: Run id string, e.g. '02-21_14-33-08'.
    """
    if now is None:
        now = time.time()
    lt = time.localtime(now)
    return time.strftime("%m-%d_%H-%M-%S", lt)


# --------------------------------------------------------------------------------------
# Args + presets
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return: Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Batch upscale images with HYPIR-SD2 using one or multiple GPUs. "
        "Optimized for large batch jobs on RunPod with network volumes."
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--batch-mode", action="store_true", default=None, help="Batch optimized mode (default).")
    mode_group.add_argument("--single-mode", action="store_true", default=None, help="Single-image/latency mode.")

    parser.add_argument(
        "--input",
        required=True,
        default=str(DEFAULT_INPUT)
        nargs="+",
        help="Input image(s): file path, directory, or glob pattern. Multiple allowed.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory root. If not set: "
        "for a directory input -> <dir>_hypir_out, "
        "for file(s) -> <common_root>/hypir_out",
    )
    parser.add_argument("--recursive", action="store_true", help="If input contains directories, scan recursively.")
    parser.add_argument("--extensions", default=DEFAULT_IMAGE_EXTENSIONS, help="Allowed image extensions (CSV).")
    parser.add_argument("--max_images", type=int, default=None, help="Process at most N images (profiling/smoke).")

    parser.add_argument("--hypir_repo_path", default=None, help="Path to the HYPIR repository.")
    parser.add_argument("--base_model_path", default=_default_base_model_path(), help="HF model id or local path for SD2.1 base.")
    parser.add_argument("--weight_path", default=str(_default_weight_path()), help="Path to HYPIR LoRA weights.")

    parser.add_argument("--hf_token", default=None, help="Hugging Face token (optional).")
    parser.add_argument("--local_files_only", action="store_true", help="Force offline mode (use local cache only).")

    parser.add_argument(
        "--pycache_prefix",
        type=str,
        default=None,
        help="Where to store Python bytecode cache (.pyc). Values: off|auto|/path. "
        "If you use compileall in the image, 'off' is usually best.",
    )

    parser.add_argument("--model_t", type=int, default=DEFAULT_MODEL_T)
    parser.add_argument("--coeff_t", type=int, default=DEFAULT_COEFF_T)
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora_modules", type=str, default=DEFAULT_LORA_MODULES)

    parser.add_argument("--scale_by", choices=["factor", "longest_side"], default=DEFAULT_SCALE_BY)
    parser.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE)
    parser.add_argument("--target_longest_side", type=int, default=DEFAULT_TARGET_LONGEST_SIDE)

    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)

    parser.add_argument("--auto_patch", dest="auto_patch", action="store_true", default=None, help="Try larger patch sizes based on VRAM, keep overlap constant.")
    parser.add_argument("--no_auto_patch", dest="auto_patch", action="store_false", help="Disable auto patch selection.")

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated visible GPU indices to use.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Use first N visible GPUs. Default: all visible.")

    parser.add_argument("--prefetch", type=int, default=None, help="How many images to prefetch per GPU worker.")
    parser.add_argument("--save_async", dest="save_async", action="store_true", default=None, help="Enable async saving.")
    parser.add_argument("--no_save_async", dest="save_async", action="store_false", help="Disable async saving.")
    parser.add_argument("--save_queue_size", type=int, default=None, help="Max queued save items per worker (keep small for huge PNGs).")
    parser.add_argument("--png_compress_level", type=int, default=DEFAULT_PNG_COMPRESS_LEVEL, help="PNG compression level (0-9).")

    parser.add_argument("--progress_file", type=str, default=DEFAULT_PROGRESS_FILE, help="Progress file path written as 'X/Y'.")
    parser.add_argument(
        "--progress_console_interval_s",
        type=float,
        default=None,
        help="How often to print 'X/Y' to stdout. 0 prints every update; negative disables.",
    )

    parser.add_argument("--warmup_models", dest="warmup_models", action="store_true", default=None, help="Warm OS page cache for model/weights.")
    parser.add_argument("--no_warmup_models", dest="warmup_models", action="store_false", help="Disable model warmup.")
    parser.add_argument("--warmup_io_workers", type=int, default=None, help="Threads used for warmup reading.")
    parser.add_argument(
        "--stage_weights",
        type=str,
        default=None,
        help="Stage weights to fast storage. Values: auto|on|off. (Batch default: on)",
    )

    parser.add_argument("--task_queue_maxsize", type=int, default=DEFAULT_TASK_QUEUE_MAXSIZE)

    parser.add_argument(
        "--mp_start_method",
        type=str,
        default=DEFAULT_MP_START_METHOD,
        choices=["auto", "spawn", "forkserver", "fork"],
        help="Multiprocessing start method. For CUDA: spawn or forkserver.",
    )

    parser.add_argument("--torch_num_threads", type=int, default=DEFAULT_TORCH_NUM_THREADS, help="0 = auto (max(1, C//G), clamped <= 16).")
    parser.add_argument("--torch_num_interop_threads", type=int, default=DEFAULT_TORCH_NUM_INTEROP_THREADS)
    parser.add_argument("--accurate_cuda_timing", action="store_true", help="Record best-effort CUDA event timing for enhance().")

    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true", default=None, help="Enable cudnn.benchmark (batch default: on).")
    parser.add_argument("--no_cudnn_benchmark", dest="cudnn_benchmark", action="store_false", help="Disable cudnn.benchmark.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 for matmul/cudnn (optional, can slightly change numerics).")

    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL)

    return parser.parse_args()


def _apply_mode_presets(args: argparse.Namespace, cpu_count: int) -> None:
    """Apply mode presets for batch-mode and single-mode.

    Only fills values when the user did not explicitly provide them.

    :param args: Parsed args (mutated in-place).
    :param cpu_count: CPU count detected on the machine.
    """
    mode = "batch"
    if args.single_mode:
        mode = "single"
    elif args.batch_mode:
        mode = "batch"

    args._mode = mode  # for logging

    if mode == "batch":
        if args.prefetch is None:
            args.prefetch = 1
        if args.save_async is None:
            args.save_async = True
        if args.save_queue_size is None:
            args.save_queue_size = 2
        if args.warmup_models is None:
            args.warmup_models = True
        if args.warmup_io_workers is None:
            args.warmup_io_workers = min(8, max(1, int(cpu_count)))
        if args.stage_weights is None:
            args.stage_weights = "on"
        if args.progress_console_interval_s is None:
            args.progress_console_interval_s = DEFAULT_PROGRESS_CONSOLE_INTERVAL_S
        if args.pycache_prefix is None:
            args.pycache_prefix = "off"
        if args.auto_patch is None:
            args.auto_patch = True
        if args.cudnn_benchmark is None:
            args.cudnn_benchmark = True

    else:
        if args.prefetch is None:
            args.prefetch = 1
        if args.save_async is None:
            args.save_async = False
        if args.save_queue_size is None:
            args.save_queue_size = 1
        if args.warmup_models is None:
            args.warmup_models = False
        if args.warmup_io_workers is None:
            args.warmup_io_workers = min(2, max(1, int(cpu_count)))
        if args.stage_weights is None:
            args.stage_weights = "off"
        if args.progress_console_interval_s is None:
            args.progress_console_interval_s = 0.0
        if args.pycache_prefix is None:
            args.pycache_prefix = "off"
        if args.auto_patch is None:
            args.auto_patch = False
        if args.cudnn_benchmark is None:
            args.cudnn_benchmark = True


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and required values.

    :param args: Parsed args.
    """
    if args.scale_by == "longest_side" and args.target_longest_side is None:
        raise SystemExit("--target_longest_side is required when scale_by=longest_side.")
    if int(args.prefetch) < 1:
        raise SystemExit("--prefetch must be >= 1.")
    if int(args.png_compress_level) < 0 or int(args.png_compress_level) > 9:
        raise SystemExit("--png_compress_level must be in [0, 9].")
    if int(args.save_queue_size) < 1:
        raise SystemExit("--save_queue_size must be >= 1.")
    if int(args.torch_num_interop_threads) < 1:
        raise SystemExit("--torch_num_interop_threads must be >= 1.")
    if int(args.patch_size) <= 0 or int(args.stride) <= 0:
        raise SystemExit("--patch_size and --stride must be > 0.")
    if int(args.stride) >= int(args.patch_size):
        raise SystemExit("--stride must be < --patch_size (overlap must be > 0).")


# --------------------------------------------------------------------------------------
# GPU selection + mp start
# --------------------------------------------------------------------------------------


def _select_gpus(args: argparse.Namespace, logger: logging.Logger) -> list[int]:
    """Select GPU indices to use.

    :param args: Parsed args.
    :param logger: Logger instance.
    :return: List of GPU indices (visible indices).
    """
    try:
        import torch
    except Exception:
        logger.warning("torch not available at selection time. Falling back to CPU.")
        return []

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU (single worker).")
        return []

    explicit_ids = _parse_csv_ints(args.gpu_ids)
    if explicit_ids:
        logger.info("Using explicit GPU ids: %s", explicit_ids)
        return explicit_ids

    n = torch.cuda.device_count()
    if args.num_gpus is None:
        ids = list(range(n))
        logger.info("Using all visible GPUs: %s", ids)
        return ids

    use_n = max(0, min(int(args.num_gpus), n))
    ids = list(range(use_n))
    logger.info("Using first %d visible GPUs: %s", use_n, ids)
    return ids


def _resolve_mp_start_method(requested: str, gpu_count: int) -> str:
    """Resolve multiprocessing start method from 'auto' and hardware.

    :param requested: Requested method (auto/spawn/forkserver/fork).
    :param gpu_count: Number of GPU workers.
    :return: Concrete method name.
    """
    req = (requested or "auto").lower().strip()
    if req != "auto":
        return req
    if int(gpu_count) >= 2:
        return "forkserver"
    return "spawn"


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()
    cpu_count = os.cpu_count() or 4
    _apply_mode_presets(args, cpu_count=int(cpu_count))
    _validate_args(args)

    run_id = _format_run_id()
    logger, file_handler, console_handler, log_file = _setup_main_logging(run_id, log_level=args.log_level)

    _apply_hf_environment(args.hf_token, args.local_files_only)
    _apply_pycache_prefix(str(args.pycache_prefix), logger=logger)

    logger.info("Run id: %s", run_id)
    logger.info("Log file: %s", log_file)
    logger.info("mode=%s cpu_count=%d", getattr(args, "_mode", "batch"), int(cpu_count))

    hypir_repo = _ensure_hypir_repo_on_path(args.hypir_repo_path)
    logger.info("HYPIR repo: %s", hypir_repo)

    images, common_root = _collect_input_images(
        inputs=args.input,
        recursive=args.recursive,
        extensions_csv=args.extensions,
        logger=logger,
    )
    if args.max_images is not None:
        images = images[: max(0, int(args.max_images))]
    logger.info("Collected %d images.", len(images))
    if common_root:
        logger.info("Common root: %s", common_root)

    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    else:
        first = Path(os.path.expanduser(args.input[0]))
        if first.exists() and first.is_dir():
            output_dir = _resolve_path(str(first.resolve()) + "_hypir_out")
        else:
            base = common_root if common_root else images[0].parent
            output_dir = (base / "hypir_out").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    weight_path = _resolve_path(args.weight_path)
    if not weight_path.exists():
        raise SystemExit(f"Weight file not found: {weight_path}")
    if not weight_path.is_file():
        raise SystemExit(f"Weight path is not a file: {weight_path}")

    base_model_path = _maybe_snapshot_download_base_model(
        base_model_path=str(args.base_model_path),
        local_files_only=bool(args.local_files_only),
        logger=logger,
    )

    staged_weight_path = _stage_file_to_fast_storage(
        src=weight_path,
        mode=str(args.stage_weights),
        logger=logger,
    )

    warmup_stats = RunningStats()
    if bool(args.warmup_models):
        warm_paths: list[Path] = [staged_weight_path]

        if _is_existing_dir(base_model_path):
            base_dir = _resolve_path(base_model_path)
            base_files = list(_iter_files_for_warmup(base_dir))
            warm_paths.extend(base_files)
            total_size = sum(p.stat().st_size for p in base_files if p.exists())
            logger.info("Base model dir: %s (files=%d, total=%s)", base_dir, len(base_files), _human_bytes(total_size))
        else:
            logger.info("Base model is not a local dir (HF id or custom). Skipping base-model warmup scan.")

        with StageTimer(warmup_stats):
            _warmup_paths(paths=warm_paths, io_workers=int(args.warmup_io_workers), logger=logger)
    else:
        logger.info("Warmup disabled.")

    gpu_ids = _select_gpus(args, logger=logger)
    num_workers = max(1, len(gpu_ids)) if gpu_ids else 1
    device_prefix = "cuda" if gpu_ids else "cpu"

    mp_method = _resolve_mp_start_method(str(args.mp_start_method), gpu_count=len(gpu_ids))
    logger.info("mp_start_method=%s (requested=%s)", mp_method, args.mp_start_method)

    import multiprocessing as mp
    ctx = mp.get_context(mp_method)

    mp_log_queue = ctx.Queue()
    listener = logging.handlers.QueueListener(mp_log_queue, file_handler, console_handler, respect_handler_level=True)
    listener.start()
    _configure_main_root_logger_to_queue(mp_log_queue, log_level=args.log_level)

    progress_stop_event = threading.Event()
    progress_thread: Optional[threading.Thread] = None

    try:
        tasks: list[Tuple[str, str]] = []
        for img in images:
            out = _build_output_path(img, output_dir=output_dir, keep_relative_to=common_root)
            tasks.append((str(img), str(out)))

        total_tasks = len(tasks)

        progress_queue = ctx.Queue()
        progress_path = _resolve_path(args.progress_file)
        _write_atomic_text(progress_path, f"0/{total_tasks}")
        progress_thread = threading.Thread(
            target=_progress_writer,
            kwargs={
                "progress_queue": progress_queue,
                "stop_event": progress_stop_event,
                "progress_path": progress_path,
                "total": total_tasks,
                "console_interval_s": float(args.progress_console_interval_s),
            },
            daemon=True,
        )
        progress_thread.start()

        torch_num_threads = int(args.torch_num_threads)
        if torch_num_threads <= 0:
            torch_num_threads = _auto_threads_per_worker(
                cpu_count=int(cpu_count),
                num_workers=int(num_workers),
                max_threads=DEFAULT_MAX_TORCH_NUM_THREADS,
            )
            logger.info(
                "Auto torch_num_threads: cpu_count=%d workers=%d -> %d (<=%d)",
                int(cpu_count),
                int(num_workers),
                int(torch_num_threads),
                int(DEFAULT_MAX_TORCH_NUM_THREADS),
            )
        torch_num_threads = _clamp_int(torch_num_threads, 1, DEFAULT_MAX_TORCH_NUM_THREADS)
        torch_num_interop_threads = _clamp_int(int(args.torch_num_interop_threads), 1, DEFAULT_MAX_TORCH_NUM_THREADS)

        if gpu_ids:
            logger.info("Starting %d GPU workers: %s", len(gpu_ids), gpu_ids)
        else:
            logger.info("Starting single CPU worker.")

        task_queue_max = int(args.task_queue_maxsize)
        tasks_queue = ctx.Queue(maxsize=task_queue_max) if task_queue_max > 0 else ctx.Queue()
        results_queue = ctx.Queue()

        for t in tasks:
            tasks_queue.put(t)
        for _ in range(num_workers):
            tasks_queue.put(None)

        cfg = WorkerConfig(
            hypir_repo_path=str(hypir_repo),
            run_id=run_id,
            base_model_path=str(base_model_path),
            weight_path=str(staged_weight_path),
            lora_modules=_parse_csv_strs(args.lora_modules),
            lora_rank=int(args.lora_rank),
            model_t=int(args.model_t),
            coeff_t=int(args.coeff_t),
            prompt=str(args.prompt),
            scale_by=str(args.scale_by),
            upscale=int(args.upscale),
            target_longest_side=(None if args.target_longest_side is None else int(args.target_longest_side)),
            patch_size=int(args.patch_size),
            stride=int(args.stride),
            auto_patch=bool(args.auto_patch),
            seed=int(args.seed),
            device_prefix=device_prefix,
            prefetch=int(args.prefetch),
            save_async=bool(args.save_async),
            png_compress_level=int(args.png_compress_level),
            save_queue_size=int(args.save_queue_size),
            torch_num_threads=int(torch_num_threads),
            torch_num_interop_threads=int(torch_num_interop_threads),
            accurate_cuda_timing=bool(args.accurate_cuda_timing),
            cudnn_benchmark=bool(args.cudnn_benchmark),
            tf32=bool(args.tf32),
        )

        start_all = time.perf_counter()
        procs: list[mp.Process] = []

        if gpu_ids:
            for gpu in gpu_ids:
                p = ctx.Process(
                    target=_worker_main,
                    kwargs={
                        "gpu_index": int(gpu),
                        "cfg": cfg,
                        "tasks_queue": tasks_queue,
                        "results_queue": results_queue,
                        "log_queue": mp_log_queue,
                        "progress_queue": progress_queue,
                        "log_level": args.log_level,
                    },
                    name=f"gpu-{gpu}",
                )
                p.start()
                procs.append(p)
        else:
            p = ctx.Process(
                target=_worker_main,
                kwargs={
                    "gpu_index": 0,
                    "cfg": cfg,
                    "tasks_queue": tasks_queue,
                    "results_queue": results_queue,
                    "log_queue": mp_log_queue,
                    "progress_queue": progress_queue,
                    "log_level": args.log_level,
                },
                name="cpu-0",
            )
            p.start()
            procs.append(p)

        worker_summaries: list[dict] = []
        for _ in range(num_workers):
            worker_summaries.append(results_queue.get())

        for p in procs:
            p.join()

        total_elapsed = time.perf_counter() - start_all

        total_processed = sum(int(s.get("processed", 0)) for s in worker_summaries)
        total_failures = sum(int(s.get("failures", 0)) for s in worker_summaries)
        ips = (total_processed / total_elapsed) if total_elapsed > 0 and total_processed > 0 else 0.0

        logger.info("========================================")
        logger.info("RUN SUMMARY")
        logger.info("images_total=%d processed_ok=%d failures=%d", len(images), total_processed, total_failures)
        logger.info("elapsed_total=%.2fs throughput=%.3f images/s", total_elapsed, ips)
        logger.info("warmup_enabled=%s warmup_time=%.2fs", bool(args.warmup_models), warmup_stats.total)
        logger.info("weight_path=%s (staged=%s)", weight_path, staged_weight_path)
        logger.info("base_model_path=%s", base_model_path)
        logger.info("pycache_prefix=%s", str(args.pycache_prefix))
        logger.info("progress_file=%s", progress_path)
        logger.info("========================================")

        for s in sorted(worker_summaries, key=lambda x: int(x.get("gpu_index", 0))):
            gi = s.get("gpu_index")
            tiling = s.get("tiling", {})
            logger.info(
                "Worker gpu=%s ok=%s fail=%s init=%.2fs enhance_mean=%.3fs total_mean=%.3fs wait_mean=%.3fs tiling=(patch=%s stride=%s overlap=%s)",
                gi,
                s.get("processed"),
                s.get("failures"),
                float(s.get("model_init", {}).get("total_s", 0.0)),
                float(s.get("enhance_wall", {}).get("mean_s", 0.0)),
                float(s.get("total_per_image", {}).get("mean_s", 0.0)),
                float(s.get("wait_for_input", {}).get("mean_s", 0.0)),
                tiling.get("patch"),
                tiling.get("stride"),
                tiling.get("overlap"),
            )

            gpu_mem = s.get("gpu_memory", {})
            if gpu_mem:
                logger.info(
                    "  gpu=%s mem_alloc=%s max_alloc=%s mem_reserved=%s max_reserved=%s",
                    gi,
                    _human_bytes(int(gpu_mem.get("memory_allocated_bytes", 0))),
                    _human_bytes(int(gpu_mem.get("max_memory_allocated_bytes", 0))),
                    _human_bytes(int(gpu_mem.get("memory_reserved_bytes", 0))),
                    _human_bytes(int(gpu_mem.get("max_memory_reserved_bytes", 0))),
                )

    finally:
        try:
            progress_stop_event.set()
        except Exception:
            pass
        try:
            if "progress_queue" in locals():
                progress_queue.put(None)
        except Exception:
            pass
        try:
            if progress_thread is not None:
                progress_thread.join(timeout=5)
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()