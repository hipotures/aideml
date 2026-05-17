"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import resource
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Callable

import humanize
from dataclasses_json import DataClassJsonMixin

from .utils.resource_monitor import ProcessResourceMonitor, ResourceSnapshot

logger = logging.getLogger("aide")


class ExecutionInterrupted(Exception):
    """Raised when the user requests immediate execution stop."""


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )
        # tb_str = "".join([l for l in tb_lines])

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue, timeout=5, log_path: Path | None = None):
        self.queue = queue
        self.timeout = timeout
        self.log_path = log_path

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            logger.warning("Queue write timed out")
        if self.log_path is not None and msg:
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(msg)
                    f.flush()
            except OSError as exc:
                logger.debug(f"Failed to write process output log: {exc}")

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        memory_limit_gb: float | None = 80.0,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            memory_limit_gb (float | None, optional): Address-space limit for the child process executing agent code. Defaults to 80 GiB.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.memory_limit_gb = memory_limit_gb
        self.output_collection_timeout = 5.0
        self.process: Process = None  # type: ignore

    def _set_child_memory_limit(self) -> None:
        if self.memory_limit_gb is None:
            return
        limit_bytes = int(float(self.memory_limit_gb) * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    def child_proc_setup(self, result_outq: Queue) -> None:
        if hasattr(os, "setpgrp"):
            os.setpgrp()
        self._set_child_memory_limit()

        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()
        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        log_path = None
        artifact_dir = os.environ.get("AIDE_NODE_ARTIFACT_DIR")
        if artifact_dir:
            try:
                log_dir = Path(artifact_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "process_stdout.log"
                log_path.touch(exist_ok=True)
            except OSError as exc:
                logger.debug(f"Failed to initialize process output log: {exc}")

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq, log_path=log_path)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {
            "__name__": "__main__",
            "__file__": self.agent_file_name,
        }
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)
            global_scope["__name__"] = "__main__"
            global_scope["__file__"] = self.agent_file_name

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in the data preview)
            os.remove(self.agent_file_name)

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        if self.process is None or self.process.pid is None:
            return False
        if not hasattr(os, "killpg") or not hasattr(os, "getpgid"):
            return False
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
            return True
        except ProcessLookupError:
            return True
        except OSError as e:
            logger.debug(f"Failed to signal process group: {e}")
            return False

    def _signal_process(self, sig: signal.Signals) -> None:
        if self.process is None:
            return
        if self._signal_process_group(sig):
            return
        try:
            if sig == signal.SIGTERM:
                self.process.terminate()
            elif sig == signal.SIGKILL:
                self.process.kill()
            elif self.process.pid is not None:
                os.kill(self.process.pid, sig)
        except ProcessLookupError:
            return

    def interrupt_execution(self) -> None:
        self.cleanup_session()

    def cleanup_session(self):
        if self.process is None:
            return
        try:
            self._signal_process(signal.SIGTERM)
            self.process.join(timeout=0.5)

            if self.process.exitcode is None:
                logger.warning("Process failed to terminate, killing immediately")
                self._signal_process(signal.SIGKILL)
                self.process.join(timeout=0.5)
        except Exception as e:
            logger.error(f"Error during process cleanup: {e}")
        finally:
            if self.process is not None:
                if self.process.exitcode is not None:
                    self.process.close()
                else:
                    logger.error("Process still running after SIGKILL; leaving unclosed")
                self.process = None

    def run(
        self,
        code: str,
        reset_session=True,
        interrupt_callback: Callable[[int], None] | None = None,
        resource_callback: Callable[[ResourceSnapshot], None] | None = None,
    ) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.
            interrupt_callback (Callable[[int], None] | None, optional): Called with the Ctrl+C count while waiting for code execution.
            resource_callback (Callable[[ResourceSnapshot], None] | None, optional): Called with process-tree resource usage snapshots while code is running.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()
        resource_monitor = None
        if resource_callback is not None and self.process.pid is not None:
            try:
                resource_monitor = ProcessResourceMonitor.from_pid(self.process.pid)
            except Exception as exc:  # noqa: BLE001 - resource UI must not stop execution
                logger.debug(f"Resource monitor unavailable: {exc}")
        if resource_monitor is not None:
            resource_callback(resource_monitor.sample())

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False
        interrupt_count = 0

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except KeyboardInterrupt:
                interrupt_count += 1
                if interrupt_callback is not None:
                    interrupt_callback(interrupt_count)
                if interrupt_count == 1:
                    continue
                self.cleanup_session()
                raise ExecutionInterrupted("Execution interrupted by user.") from None
            except queue.Empty:
                if resource_monitor is not None and resource_callback is not None:
                    resource_callback(resource_monitor.sample())

                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    logger.warning(f"Execution exceeded timeout of {self.timeout}s")
                    os.kill(self.process.pid, signal.SIGINT)
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    if running_time > self.timeout + 5:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                # Add 5-second timeout for output collection
                if time.time() - start_collect > self.output_collection_timeout:
                    logger.warning("Output collection timed out")
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        received_eof = bool(output and output[-1] == "<|EOF|>")
        if received_eof:
            output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]
        if not received_eof:
            msg = "REPL output stream ended before EOF marker"
            output.append(f"RuntimeError: {msg}\n")
            if e_cls_name is None:
                e_cls_name = "RuntimeError"
                exc_info = {"args": [msg]}
                exc_stack = None

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
