"""
machine.py  –  Serial worker + all GRBL communication.
Every significant event is timestamped and tagged with thread name.
"""

import time
import threading
import queue

import serial
import serial.tools.list_ports

from config import PEN_DOWN_Z as _DEFAULT_PEN_DOWN_Z, JOG_FEED, DRAW_FEED

BAUD          = 115200
READ_TIMEOUT  = 2.0
POLL_INTERVAL = 0.5

PEN_UP_Z  = 0

PAGE_X_MM = 210.0
PAGE_Y_MM = 300.0
MARGIN_MM =  10.0

DRAW_W = PAGE_X_MM - 2 * MARGIN_MM
DRAW_H = PAGE_Y_MM - 2 * MARGIN_MM

_log_lock = threading.Lock()

def _log(tag, msg):
    t = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
    thread = threading.current_thread().name
    with _log_lock:
        print(f"[{t}][{thread}][{tag}] {msg}", flush=True)


def list_serial_ports() -> list:
    ports = serial.tools.list_ports.comports()
    return [p.device for p in sorted(ports)]


def draw_to_machine(x: float, y: float) -> tuple:
    x_mm = x + MARGIN_MM
    y_mm = y + MARGIN_MM
    x_mm = max(MARGIN_MM, min(PAGE_X_MM - MARGIN_MM, x_mm))
    y_mm = max(MARGIN_MM, min(PAGE_Y_MM - MARGIN_MM, y_mm))
    return x_mm, y_mm


class Machine:
    def __init__(self):
        self._ser = None
        self._cmd_queue = queue.Queue()
        self._worker_thread = None
        self._running = False

        self._pen_down_z: float = _DEFAULT_PEN_DOWN_Z
        self._pen_is_down: bool = False
        self.jog_step_mm: float = 5.0

        self._rt_lock = threading.Lock()
        self._abort = threading.Event()
        self._in_cleanup = False
        self._home_mpos = None   # MPos recorded at home position, used to restore G92 after reset

        self._status_lock = threading.Lock()
        self.status = {
            "state":    "Disconnected",
            "x":        0.0,
            "y":        0.0,
            "z":        0.0,
            "pen_down": False,
        }

        self.pause_event = threading.Event()
        self.pause_event.set()

    # ── public API ─────────────────────────────────────────────────────────────

    def connect(self, port: str) -> None:
        _log("CONNECT", f"Opening {port}")
        if self._running:
            _log("CONNECT", "Already running, skipping")
            return
        self._ser = serial.Serial(port, baudrate=BAUD, timeout=READ_TIMEOUT)
        time.sleep(0.5)
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, name="MachineWorker", daemon=True
        )
        self._worker_thread.start()
        self._cmd_queue.put(("INIT",))
        _log("CONNECT", "Worker started, INIT queued")

    def disconnect(self) -> None:
        _log("DISCONNECT", "Disconnecting")
        if not self._running:
            return
        self._send_realtime(b"!")
        self._cmd_queue.put(("SHUTDOWN",))
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._running = False
        self._update_status(state="Disconnected")
        _log("DISCONNECT", "Done")

    def set_pen_depth(self, z_mm: float) -> None:
        self._pen_down_z = float(z_mm)

    def pen_up(self) -> None:
        _log("API", "pen_up queued")
        self._cmd_queue.put(("PEN_UP",))

    def pen_down(self) -> None:
        _log("API", "pen_down queued")
        self._cmd_queue.put(("PEN_DOWN",))

    def home(self) -> None:
        _log("API", "HOME queued")
        self._cmd_queue.put(("HOME",))

    def set_origin(self) -> None:
        _log("API", "SET_ORIGIN queued")
        self._cmd_queue.put(("SET_ORIGIN",))

    def jog(self, dx: float, dy: float) -> None:
        # Discard any pending jog commands — only keep the latest
        while not self._cmd_queue.empty():
            try:
                peeked = self._cmd_queue.queue[0]
                if peeked[0] == "JOG":
                    self._cmd_queue.get_nowait()
                else:
                    break
            except (IndexError, queue.Empty):
                break
        _log("API", f"JOG queued dx={dx} dy={dy}")
        self._cmd_queue.put(("JOG", dx, dy))

    def send_gcode(self, line: str) -> None:
        self._cmd_queue.put(("GCODE", line))

    def send_raw(self, cmd: str) -> list:
        if not self._ser or not self._ser.is_open:
            return ["Not connected"]
        try:
            self._ser.reset_input_buffer()
            self._ser.write((cmd + "\n").encode("ascii"))
            self._ser.flush()
            lines = []
            deadline = time.time() + 3.0
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                lines.append(line)
                if line.startswith("ok") or line.startswith("error"):
                    break
            return lines if lines else ["(no response)"]
        except Exception as e:
            return [f"Error: {e}"]

    def pause(self) -> None:
        _log("PAUSE", "pause() called — clearing pause_event, setting abort, sending !")
        self.pause_event.clear()
        self._abort.set()
        self._flush_queue()
        self._send_realtime(b"!")
        self._update_status(state="Hold")
        _log("PAUSE", f"pause() done — queue size now: {self._cmd_queue.qsize()}")

    def resume(self) -> None:
        _log("RESUME", "resume() called — clearing abort, sending ~, setting pause_event")
        self._abort.clear()
        self._send_realtime(b"~")
        self._update_status(state="Running")
        self.pause_event.set()
        _log("RESUME", "resume() done")

    def stop(self) -> None:
        _log("STOP", "stop() called — setting abort, queuing STOP_CLEANUP")
        self.pause_event.clear()
        self._abort.set()
        self._flush_queue()               # clear model commands
        _log("STOP", f"queue size before STOP_CLEANUP: {self._cmd_queue.qsize()}")
        self._cmd_queue.put(("STOP_CLEANUP",))
        _log("STOP", "STOP_CLEANUP queued")

    def is_connected(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        with self._status_lock:
            return dict(self.status)

    def flush_queue(self) -> None:
        before = self._cmd_queue.qsize()
        self._flush_queue()
        after = self._cmd_queue.qsize()
        _log("FLUSH", f"Flushed queue: {before} → {after} items")

    # ── internal ───────────────────────────────────────────────────────────────

    def _send_realtime(self, byte: bytes) -> None:
        name = {b"!": "FEED_HOLD", b"~": "CYCLE_RESUME", b"\x18": "SOFT_RESET"}.get(byte, repr(byte))
        _log("REALTIME", f"Sending {name}")
        if self._ser and self._ser.is_open:
            with self._rt_lock:
                self._ser.write(byte)
                self._ser.flush()
        _log("REALTIME", f"{name} sent")

    def _flush_queue(self) -> None:
        while not self._cmd_queue.empty():
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break

    def _update_status(self, **kwargs) -> None:
        with self._status_lock:
            self.status.update(kwargs)

    def _send_and_wait_ok(self, cmd: str) -> None:
        _log("SERIAL", f">> {cmd}")
        self._ser.write((cmd + "\n").encode("ascii"))
        self._ser.flush()
        while True:
            if self._abort.is_set():
                _log("SERIAL", f"ABORT detected while waiting for ok on: {cmd}")
                raise RuntimeError("Aborted")
            line = self._ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            _log("SERIAL", f"<< {line}")
            if line.startswith("ok"):
                return
            if line.startswith("error") or "ALARM" in line:
                raise RuntimeError(f"GRBL error: {line}")

    def _soft_reset(self) -> None:
        self._send_realtime(b"\x18")

    def _wait_for_ready(self, timeout: float = 5.0) -> None:
        _log("INIT", "Waiting for GRBL ready banner...")
        start = time.time()
        while True:
            line = self._ser.readline().decode(errors="ignore").strip()
            if line:
                _log("INIT", f"<< {line}")
            if "Grbl" in line:
                _log("INIT", "GRBL ready banner received")
                return
            if time.time() - start > timeout:
                raise TimeoutError("GRBL startup timeout")

    def _poll_status(self) -> None:
        try:
            self._ser.write(b"?\n")
            self._ser.flush()
            deadline = time.time() + 1.0
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="ignore").strip()
                if not line or not line.startswith("<"):
                    continue
                parts = line.strip("<>").split("|")
                state = parts[0]
                if not state:
                    return
                x = y = z = None
                for part in parts[1:]:
                    if part.startswith("WPos:") or part.startswith("MPos:"):
                        coords = part[5:].split(",")
                        if len(coords) >= 3:
                            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                if x is not None:
                    self._update_status(state=state, x=x, y=y, z=z)
                return
        except Exception as e:
            _log("POLL", f"Error: {e}")

    def _do_pen_up(self) -> None:
        _log("PEN", "Pen UP")
        self._send_and_wait_ok(f"G0 Z{PEN_UP_Z:.3f}")
        self._pen_is_down = False
        self._update_status(pen_down=False)

    def _do_pen_down(self) -> None:
        _log("PEN", "Pen DOWN")
        self._send_and_wait_ok(f"G0 Z{self._pen_down_z:.3f}")
        self._pen_is_down = True
        self._update_status(pen_down=True)

    # ── worker thread ──────────────────────────────────────────────────────────

    def _worker(self) -> None:
        _log("WORKER", "Worker thread started")
        last_poll = 0.0

        while self._running:
            now = time.time()
            if now - last_poll > POLL_INTERVAL and self._ser and self._ser.is_open and not self._in_cleanup:
                self._poll_status()
                last_poll = time.time()

            try:
                cmd = self._cmd_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            tag = cmd[0]
            _log("WORKER", f"Processing command: {tag}  (queue size after get: {self._cmd_queue.qsize()})")

            try:
                if tag == "INIT":
                    _log("WORKER", "INIT: soft reset + wait for ready")
                    self._soft_reset()
                    self._wait_for_ready()
                    self._ser.reset_input_buffer()
                    # Poll status — wait for a proper '<...>' response line
                    self._ser.write(b"?\n")
                    self._ser.flush()
                    grbl_state = ""
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        resp = self._ser.readline().decode(errors="ignore").strip()
                        if resp.startswith("<"):
                            grbl_state = resp
                            break
                    _log("WORKER", f"INIT: GRBL state: {grbl_state}")
                    if "Alarm" in grbl_state:
                        _log("WORKER", "INIT: ALARM detected, sending $X")
                        self._send_and_wait_ok("$X")
                        self._ser.reset_input_buffer()
                    self._send_and_wait_ok("G21")
                    self._send_and_wait_ok("G90")
                    self._send_and_wait_ok("G94")
                    self._update_status(state="Idle")
                    _log("WORKER", "INIT complete")

                elif tag == "GCODE":
                    self._send_and_wait_ok(cmd[1])

                elif tag == "PEN_UP":
                    self._do_pen_up()

                elif tag == "PEN_DOWN":
                    self._do_pen_down()

                elif tag == "HOME":
                    _log("WORKER", "HOME: running $H")
                    self._update_status(state="Homing")
                    self._send_and_wait_ok("$H")
                    self._send_and_wait_ok("G92 X0 Y0 Z0")
                    # Home MPos is always (-PAGE_X_MM, -PAGE_Y_MM) based on $130/$131
                    # Store it so STOP_CLEANUP can restore G92 after a soft reset
                    self._home_mpos = (-PAGE_X_MM, -PAGE_Y_MM)
                    _log("WORKER", f"HOME: home MPos set to {self._home_mpos}")
                    self._update_status(state="Idle", x=0.0, y=0.0, z=0.0)
                    _log("WORKER", "HOME complete")

                elif tag == "SET_ORIGIN":
                    self._send_and_wait_ok("G92 X0 Y0 Z0")

                elif tag == "JOG":
                    _, dx, dy = cmd
                    _log("WORKER", f"JOG: dx={dx} dy={dy}  abort={self._abort.is_set()}")
                    self._send_and_wait_ok(f"$J=G21G91 X{dx:.3f} Y{dy:.3f} F{JOG_FEED}")

                elif tag == "STOP_CLEANUP":
                    self._in_cleanup = True
                    self._abort.clear()
                    _log("WORKER", "STOP_CLEANUP: soft reset")
                    self._send_realtime(b"\x18")
                    self._ser.reset_input_buffer()
                    try:
                        self._wait_for_ready(timeout=5.0)
                    except TimeoutError:
                        _log("WORKER", "STOP_CLEANUP: timed out waiting for GRBL ready")
                    self._ser.reset_input_buffer()
                    # Check and clear alarm
                    grbl_state = ""
                    self._ser.write(b"?\n")
                    self._ser.flush()
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        resp = self._ser.readline().decode(errors="ignore").strip()
                        if resp.startswith("<"):
                            grbl_state = resp
                            break
                    _log("WORKER", f"STOP_CLEANUP: state after reset: {grbl_state}")
                    if "Alarm" in grbl_state:
                        _log("WORKER", "STOP_CLEANUP: clearing alarm with $X")
                        self._send_and_wait_ok("$X")
                        self._ser.reset_input_buffer()
                    # Reinit modes
                    self._send_and_wait_ok("G21")
                    self._send_and_wait_ok("G90")
                    self._send_and_wait_ok("G94")
                    # Restore X/Y work coordinates from home MPos
                    cur_mpos_x = cur_mpos_y = 0.0
                    for part in grbl_state.strip("<>").split("|"):
                        if part.startswith("MPos:"):
                            coords = part[5:].split(",")
                            if len(coords) >= 2:
                                cur_mpos_x = float(coords[0])
                                cur_mpos_y = float(coords[1])
                    if self._home_mpos is not None:
                        home_x, home_y = self._home_mpos
                        wpos_x = cur_mpos_x - home_x
                        wpos_y = cur_mpos_y - home_y
                    else:
                        wpos_x = wpos_y = 0.0
                    _log("WORKER", f"STOP_CLEANUP: restoring WPos X={wpos_x:.3f} Y={wpos_y:.3f}")
                    self._send_and_wait_ok(f"G92 X{wpos_x:.3f} Y{wpos_y:.3f}")
                    # Reset Z machine position to 0 using custom firmware command.
                    # After soft reset the spring returns pen to physical Z=0,
                    # $Z tells GRBL to set MPos Z=0 to match physical reality.
                    self._send_and_wait_ok("$Z")
                    _log("WORKER", "STOP_CLEANUP: Z machine position reset to 0")
                    self._pen_is_down = False
                    self._update_status(pen_down=False, state="Idle")
                    self.pause_event.set()
                    self._in_cleanup = False
                    _log("WORKER", "STOP_CLEANUP complete — machine ready")

                elif tag == "SHUTDOWN":
                    _log("WORKER", "SHUTDOWN received")
                    try:
                        self._abort.clear()
                        self._send_realtime(b"~")
                        time.sleep(0.1)
                        self._do_pen_up()
                    except Exception as e:
                        _log("WORKER", f"SHUTDOWN pen up failed: {e}")
                    self._running = False
                    if self._ser and self._ser.is_open:
                        self._ser.close()
                    _log("WORKER", "SHUTDOWN complete")
                    return

            except Exception as e:
                _log("WORKER", f"ERROR on {tag}: {e}")
                self._update_status(state=f"Err: {e}")
                self._abort.clear()
                self._in_cleanup = False
                if "ALARM" in str(e):
                    _log("WORKER", "Auto-unlocking ALARM with $X")
                    try:
                        self._ser.reset_input_buffer()
                        self._send_and_wait_ok("$X")
                        self._ser.reset_input_buffer()
                        self._update_status(state="Idle")
                    except Exception as e2:
                        _log("WORKER", f"Auto-unlock failed: {e2}")

        _log("WORKER", "Worker thread exiting")
