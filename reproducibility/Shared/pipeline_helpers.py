# Small dbutils stand ins for local runs.
# this lets the databricks notebooks run as plain python during checks

from dataclasses import dataclass
import os
from pathlib import Path
import runpy
import sys
from typing import Iterable

REPRO_DIR_NAMES = ("reproducibility", "Thesis_Reproducibility_Code")


class _LocalString(str):
    def get(self) -> str:
        return str(self)


class _LocalNotebookContext:
    def __init__(self, notebook_path: str = ""):
        self._notebook_path = notebook_path

    def notebookPath(self) -> _LocalString:
        return _LocalString(self._notebook_path)

    def apiToken(self) -> _LocalString:
        return _LocalString(os.environ.get("QWEN3_API_TOKEN", os.environ.get("DATABRICKS_TOKEN", "")))


class _LocalNotebook:
    def __init__(self, dbutils_obj: "_LocalDbutils", notebook_path: str):
        self.entry_point = self
        self._dbutils_obj = dbutils_obj
        self._notebook_path = notebook_path

    def __call__(self) -> "_LocalNotebook":
        return self

    def getDbutils(self) -> "_LocalDbutils":
        return self._dbutils_obj

    def getContext(self) -> _LocalNotebookContext:
        return _LocalNotebookContext(self._notebook_path)

    def run(self, path: str, timeout_seconds: int, params: dict[str, str] | None = None) -> str:
        run_workspace_notebook(path, params or {}, timeout_seconds=timeout_seconds, dry_run=False)
        return "OK"

    def exit(self, value: str = "") -> None:
        raise SystemExit(value)


class _LocalWidgets:
    def __init__(self, params: dict[str, str]):
        self._params = params

    def get(self, name: str) -> str:
        return self._params.get(name, os.environ.get(name, ""))


class _LocalDbutils:
    class _Library:
        def restartPython(self) -> None:
            return None

    def __init__(self, params: dict[str, str] | None = None, notebook_path: str = ""):
        self.widgets = _LocalWidgets(params or {})
        self.library = self._Library()
        self.notebook = _LocalNotebook(self, notebook_path)


class _LocalSparkConf:
    def get(self, name: str, default: str | None = None) -> str:
        if name == "spark.databricks.workspaceUrl":
            value = os.environ.get("DATABRICKS_WORKSPACE_URL", "")
            if value:
                return value
        if default is not None:
            return default
        raise KeyError(name)


class _LocalSpark:
    conf = _LocalSparkConf()


def get_dbutils():
    try:
        return dbutils  # type: ignore[name-defined]
    except NameError:
        dbutils_obj = None
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and "dbutils" in shell.user_ns:
            return shell.user_ns["dbutils"]
    except Exception:
        shell = None
    raise NameError("Databricks dbutils is not available in this execution context")


def get_dbutils_or_none():
    try:
        return get_dbutils()
    except NameError:
        return None


def widget_or_default(name: str, default: str) -> str:
    try:
        value = get_dbutils().widgets.get(name)
        if value not in (None, ""):
            return value
    except Exception:
        value = ""
    return os.environ.get(name, default)


def bool_widget(name: str, default: bool = False) -> bool:
    return widget_or_default(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def current_notebook_path() -> str:
    try:
        return get_dbutils().notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    except Exception:
        return ""


def workspace_root() -> str:
    override = widget_or_default("THESIS_WORKSPACE_ROOT", os.environ.get("THESIS_WORKSPACE_ROOT", "")).strip()
    if override:
        return override.rstrip("/")
    notebook_path = current_notebook_path()
    normalised_notebook_path = notebook_path.replace("\\", "/")
    for dirname in REPRO_DIR_NAMES:
        marker = f"/{dirname}/"
        if marker in normalised_notebook_path:
            return normalised_notebook_path.split(marker, 1)[0].rstrip("/")
    if "__file__" in globals():
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            if parent.name in REPRO_DIR_NAMES:
                return str(parent.parent)
    fallback = os.environ.get("THESIS_WORKSPACE_ROOT", "").strip()
    assert fallback, "Set THESIS_WORKSPACE_ROOT"
    return fallback.rstrip("/")


def reproduction_code_dir(workspace_root_value: str | None = None) -> str:
    """Return the reproduction-code root in local or Databricks layouts."""
    root = Path(workspace_root_value or workspace_root())
    for dirname in REPRO_DIR_NAMES:
        candidate = root / dirname
        if candidate.exists():
            return str(candidate)
    return f"{str(root).rstrip('/')}/Thesis_Reproducibility_Code"


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def thesis_data_root() -> Path:
    override = widget_or_default("THESIS_DATA_ROOT", os.environ.get("THESIS_DATA_ROOT", "")).strip()
    if override:
        return Path(override)
    if Path("/dbfs").exists():
        return Path(widget_or_default("THESIS_DBFS_ROOT", "/dbfs/FileStore/thesis"))
    if "__file__" in globals():
        return _find_repo_root(Path(__file__).resolve())
    return Path.cwd()


@dataclass(frozen=True)
class PipelineTask:
    key: str
    notebook: str
    expected_outputs: tuple[str, ...]
    analysis_root: str
    expected_data_outputs: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return f"{self.analysis_root}/{self.notebook}"


def selected_tasks(tasks: Iterable[PipelineTask], task_filter: str, pipeline_name: str) -> list[PipelineTask]:
    if not task_filter:
        return list(tasks)
    requested = {item.strip() for item in task_filter.split(",") if item.strip()}
    selected = [task for task in tasks if task.key in requested]
    missing = requested.difference({task.key for task in selected})
    assert not missing, f"Unknown {pipeline_name} task key(s): {sorted(missing)}"
    return selected


def _resolve_output_path(root: Path, output: str) -> Path:
    path = Path(output)
    return path if path.is_absolute() else root / path


def _notebook_to_local_path(path: str) -> Path:
    raw = Path(path)
    candidates = [raw]
    if raw.suffix != ".py":
        candidates.append(raw.with_suffix(".py"))
    path_text = str(path)
    if path_text.startswith("/Workspace/"):
        stripped = Path(path_text[len("/Workspace/") :])
        candidates.extend([stripped, stripped.with_suffix(".py")])
    matches = [candidate for candidate in candidates if candidate.exists()]
    assert matches, f"Could not resolve notebook path: {path}"
    return matches[0]


def _run_local_notebook(path: str, params: dict[str, str]) -> None:
    local_path = _notebook_to_local_path(path)
    old_env = {key: os.environ.get(key) for key in params}
    old_path = list(sys.path)
    try:
        for key, value in params.items():
            os.environ[key] = str(value)
        shared_dir = local_path.parents[2] / "Shared" if len(local_path.parents) >= 3 else None
        if shared_dir is not None and shared_dir.exists():
            shared_text = str(shared_dir)
            if shared_text not in sys.path:
                sys.path.append(shared_text)
        runpy.run_path(
            str(local_path),
            run_name="__main__",
            init_globals={
                "dbutils": _LocalDbutils(params, str(local_path)),
                "spark": _LocalSpark(),
            },
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.path[:] = old_path


def verify_task_outputs(task: PipelineTask, figure_root: Path, figure_subdir: str, data_root: Path) -> None:
    missing = [
        str(figure_root / figure_subdir / filename)
        for filename in task.expected_outputs
        if not (figure_root / figure_subdir / filename).exists()
    ]
    missing.extend(
        str(_resolve_output_path(data_root, filename))
        for filename in task.expected_data_outputs
        if not _resolve_output_path(data_root, filename).exists()
    )
    assert not (missing), f"{task.key} did not produce expected output(s): {missing}"


def run_workspace_notebook(
    path: str,
    params: dict[str, str],
    timeout_seconds: int = 0,
    dry_run: bool = False,
) -> None:
    print("=" * 88)
    print(f"Notebook: {path}")
    print(f"Parameters: {params}")
    if dry_run:
        return
    dbutils_obj = get_dbutils_or_none()
    if dbutils_obj is None:
        _run_local_notebook(path, params)
        print("Notebook result: OK")
    else:
        result = dbutils_obj.notebook.run(path, timeout_seconds, params)
        print(f"Notebook result: {result}")


def run_pipeline_task(
    task: PipelineTask,
    run_params: dict[str, str],
    figure_root: Path,
    figure_subdir: str,
    data_root: Path,
    timeout_seconds: int = 0,
    dry_run: bool = False,
    verify_outputs: bool = True,
) -> None:
    print("=" * 88)
    print(f"Task: {task.key}")
    print(f"Notebook: {task.path}")
    print(f"Figure folder: {figure_root / figure_subdir}")
    if task.expected_data_outputs:
        print(f"Data outputs: {list(task.expected_data_outputs)}")
    if dry_run:
        return
    dbutils_obj = get_dbutils_or_none()
    if dbutils_obj is None:
        _run_local_notebook(task.path, run_params)
        print("Notebook result: OK")
    else:
        result = dbutils_obj.notebook.run(task.path, timeout_seconds, run_params)
        print(f"Notebook result: {result}")
    if verify_outputs:
        verify_task_outputs(task, figure_root, figure_subdir, data_root)
        print("Outputs found.")


def run_notebook_pipeline(
    pipeline_name: str,
    tasks: Iterable[PipelineTask],
    task_filter: str,
    run_params: dict[str, str],
    figure_root: Path,
    figure_subdir: str,
    data_root: Path,
    timeout_seconds: int = 0,
    dry_run: bool = False,
    verify_outputs: bool = True,
) -> None:
    selected = selected_tasks(tasks, task_filter, pipeline_name)
    print(f"{pipeline_name} figure pipeline")
    print(f"Tasks: {[task.key for task in selected]}")
    print(f"Parameters: {run_params}")
    print(f"Dry run: {dry_run}; verify outputs: {verify_outputs}")
    for task in selected:
        run_pipeline_task(
            task,
            run_params,
            figure_root,
            figure_subdir,
            data_root,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            verify_outputs=verify_outputs,
        )
    print("=" * 88)
    print(f"{pipeline_name} complete.")
