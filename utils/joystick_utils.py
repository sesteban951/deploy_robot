##
#
# Joystick math functions. Tested with XBOX controllers.
# Also includes a userspace XInput USB reader (XInputUSBJoystick) for the Jetson Orin.
#
##

# standard imports
import ctypes
import errno
import fcntl
import glob
import os
import struct
import threading
import time
from dataclasses import dataclass

#######################################################################
# JOYSTICK STATE
#######################################################################

# joystick state dataclass
@dataclass
class JoystickState:
    
    # Buttons
    A: int = 0
    B: int = 0
    X: int = 0
    Y: int = 0
    LB: int = 0
    RB: int = 0
    LMB: int = 0   # left middle button
    RMB: int = 0   # right middle button
    LS: int = 0    # left stick press
    RS: int = 0    # right stick press
    XBOX: int = 0  # XBOX button in the middle of the controller

    # Axes
    LS_X: float = 0.0
    LS_Y: float = 0.0
    RS_X: float = 0.0
    RS_Y: float = 0.0
    LT: float = 0.0
    RT: float = 0.0
    L_DPAD: float = 0.0  # D-PAD left
    R_DPAD: float = 0.0  # D-PAD right
    U_DPAD: float = 0.0  # D-PAD up
    D_DPAD: float = 0.0  # D-PAD down

#######################################################################
# Helper Functions
#######################################################################

# pygame joystick to JoystickState conversion
def pygame_to_joystick_state(joystick):

    state = JoystickState()

    # joysticks
    state.LS_X = -joystick.get_axis(0) # invert x-axis
    state.LS_Y = -joystick.get_axis(1) # invert y-axis
    state.RS_X = -joystick.get_axis(3) # invert x-axis
    state.RS_Y = -joystick.get_axis(4) # invert y-axis

    # triggers
    state.LT = 0.5 * joystick.get_axis(2) + 0.5
    state.RT = 0.5 * joystick.get_axis(5) + 0.5

    # D-PAD
    DPAD = joystick.get_hat(0)
    
    DPAD_X = DPAD[0]
    if DPAD_X <= -0.5:
        state.L_DPAD = 1.0
        state.R_DPAD = 0.0
    elif DPAD_X >= 0.5:
        state.L_DPAD = 0.0
        state.R_DPAD = 1.0
    else:
        state.L_DPAD = 0.0
        state.R_DPAD = 0.0

    DPAD_Y = DPAD[1]
    if DPAD_Y <= -0.5:
        state.U_DPAD = 0.0
        state.D_DPAD = 1.0
    elif DPAD_Y >= 0.5:
        state.U_DPAD = 1.0
        state.D_DPAD = 0.0
    else:
        state.U_DPAD = 0.0
        state.D_DPAD = 0.0

    # buttons
    state.A = joystick.get_button(0)
    state.B = joystick.get_button(1)
    state.X = joystick.get_button(2)
    state.Y = joystick.get_button(3)
    state.LB = joystick.get_button(4)
    state.RB = joystick.get_button(5)
    state.LMB = joystick.get_button(6)
    state.RMB = joystick.get_button(7)
    state.XBOX = joystick.get_button(8)
    state.LS = joystick.get_button(9)
    state.RS = joystick.get_button(10)

    return state


# ROS Joy message to JoystickState conversion
def rosjoy_to_joystick_state(joy_msg):

    state = JoystickState()

    # joysticks
    state.LS_X = joy_msg.axes[0]
    state.LS_Y = joy_msg.axes[1]
    state.RS_X = joy_msg.axes[3]
    state.RS_Y = joy_msg.axes[4]

    # triggers
    state.LT = -0.5 * joy_msg.axes[2] + 0.5
    state.RT = -0.5 * joy_msg.axes[5] + 0.5

    # D-PAD axes
    DPAD_X = joy_msg.axes[6]
    if DPAD_X == 1.0:
        state.L_DPAD = 1.0
        state.R_DPAD = 0.0
    elif DPAD_X == -1.0:
        state.L_DPAD = 0.0
        state.R_DPAD = 1.0
    else:
        state.L_DPAD = 0.0
        state.R_DPAD = 0.0

    DPAD_Y = joy_msg.axes[7]
    if DPAD_Y == -1.0:
        state.U_DPAD = 0.0
        state.D_DPAD = 1.0
    elif DPAD_Y == 1.0:
        state.U_DPAD = 1.0
        state.D_DPAD = 0.0
    else:
        state.U_DPAD = 0.0
        state.D_DPAD = 0.0

    # buttons
    state.A = joy_msg.buttons[0]
    state.B = joy_msg.buttons[1]
    state.X = joy_msg.buttons[2]
    state.Y = joy_msg.buttons[3]
    state.LB = joy_msg.buttons[4]
    state.RB = joy_msg.buttons[5]
    state.LMB = joy_msg.buttons[6]
    state.RMB = joy_msg.buttons[7]
    state.XBOX = joy_msg.buttons[8]
    state.LS = joy_msg.buttons[9]
    state.RS = joy_msg.buttons[10]

    return state

#######################################################################
# XInput over USB (userspace driver)
#######################################################################
#
# Reads an XInput (Xbox 360 protocol) gamepad directly over usbfs, standard library only.
#
# Why: on the Jetson Orin (JetPack 5, kernel 5.10) the xpad driver doesn't know the
# 8BitDo Ultimate 2C (2dc8:310a) and never sends the "set LED" packet this controller
# waits for before streaming input, so joy_node / pygame see no events.
#
# Needs read/write access to the controller's /dev/bus/usb node, e.g. a udev rule:
#   SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="2dc8", ATTR{idProduct}=="310a", MODE="0666"

# XInput controllers to look for: (vendor, product)
XINPUT_DEVICES = [("2dc8", "310a")]   # 8BitDo Ultimate 2C Wireless (XInput mode)
_XINPUT_INTF, _EP_IN, _EP_OUT = 0, 0x84, 0x05


def _IOC(d, t, nr, size):
    return (d << 30) | (size << 16) | (ord(t) << 8) | nr

class _CtrlTransfer(ctypes.Structure):
    _fields_ = [("bRequestType", ctypes.c_uint8), ("bRequest", ctypes.c_uint8),
                ("wValue", ctypes.c_uint16), ("wIndex", ctypes.c_uint16),
                ("wLength", ctypes.c_uint16), ("timeout", ctypes.c_uint32),
                ("data", ctypes.c_void_p)]

class _BulkTransfer(ctypes.Structure):  # also valid for interrupt endpoints
    _fields_ = [("ep", ctypes.c_uint), ("len", ctypes.c_uint),
                ("timeout", ctypes.c_uint), ("data", ctypes.c_void_p)]

_USBDEVFS_CONTROL = _IOC(3, "U", 0, ctypes.sizeof(_CtrlTransfer))
_USBDEVFS_BULK = _IOC(3, "U", 2, ctypes.sizeof(_BulkTransfer))
_USBDEVFS_CLAIMINTERFACE = _IOC(2, "U", 15, 4)
_USBDEVFS_RELEASEINTERFACE = _IOC(2, "U", 16, 4)

# Xbox 360 report button bits
_XBTN = dict(DUP=0, DDOWN=1, DLEFT=2, DRIGHT=3, START=4, BACK=5, LS=6, RS=7,
             LB=8, RB=9, XBOX=10, A=12, B=13, X=14, Y=15)


def _find_xinput_device():
    """Return (sysfs_path, /dev/bus/usb node) of the first known XInput controller, or None."""
    for dev in glob.glob("/sys/bus/usb/devices/*"):
        try:
            with open(f"{dev}/idVendor") as f:
                vid = f.read().strip()
            with open(f"{dev}/idProduct") as f:
                pid = f.read().strip()
            if (vid, pid) not in XINPUT_DEVICES:
                continue
            with open(f"{dev}/busnum") as f:
                bus = int(f.read())
            with open(f"{dev}/devnum") as f:
                num = int(f.read())
            return dev, f"/dev/bus/usb/{bus:03d}/{num:03d}"
        except (OSError, ValueError):
            continue
    return None


class _XInputUSBDevice:
    """Minimal usbfs XInput (Xbox 360 protocol) device handle."""

    def __init__(self, node_path: str):
        self.fd = os.open(node_path, os.O_RDWR)
        try:
            fcntl.ioctl(self.fd, _USBDEVFS_CLAIMINTERFACE, struct.pack("I", _XINPUT_INTF))
        except OSError:
            os.close(self.fd)
            raise
        # init vendor request (what newer kernels' xpad sends; harmless if ignored)
        try:
            buf = ctypes.create_string_buffer(20)
            x = _CtrlTransfer(0xC1, 0x01, 0x0100, 0x0000, 20, 500, ctypes.cast(buf, ctypes.c_void_p))
            fcntl.ioctl(self.fd, _USBDEVFS_CONTROL, x)
        except OSError:
            pass
        # "set LED" packet (player 1): the controller only streams input after this
        self._transfer(_EP_OUT, bytes([0x01, 0x03, 0x02]), 500)

    def _transfer(self, ep, data, timeout_ms):
        if isinstance(data, int):
            buf, n = ctypes.create_string_buffer(data), data
        else:
            buf, n = ctypes.create_string_buffer(data, len(data)), len(data)
        x = _BulkTransfer(ep, n, timeout_ms, ctypes.cast(buf, ctypes.c_void_p))
        got = fcntl.ioctl(self.fd, _USBDEVFS_BULK, x)
        return buf.raw[:got]

    def read_report(self, timeout_ms=100):
        """Return a 20-byte input report, or None on timeout / non-input packet."""
        try:
            d = self._transfer(_EP_IN, 64, timeout_ms)
        except OSError as e:
            if e.errno == errno.ETIMEDOUT:
                return None
            raise
        if len(d) >= 14 and d[0] == 0x00 and d[1] == 0x14:
            return d
        return None

    def close(self):
        try:
            fcntl.ioctl(self.fd, _USBDEVFS_RELEASEINTERFACE, struct.pack("I", _XINPUT_INTF))
        except OSError:
            pass
        os.close(self.fd)


def _apply_deadzone(v, deadzone):
    # same scheme as joy_node: zero inside the deadzone, rescale outside to keep [-1, 1]
    if abs(v) < deadzone:
        return 0.0
    scale = 1.0 / (1.0 - deadzone)
    return (v - deadzone) * scale if v > 0 else (v + deadzone) * scale


def _stick(v):
    return v / 32768.0 if v < 0 else v / 32767.0


# XInput USB report to JoystickState conversion (same conventions as rosjoy_to_joystick_state)
def xinput_report_to_joystick_state(report, deadzone=0.05):

    state = JoystickState()

    btn = report[2] | (report[3] << 8)
    bit = lambda name: (btn >> _XBTN[name]) & 1
    lx, ly, rx, ry = struct.unpack_from("<hhhh", report, 6)   # XInput: right +, up +

    # joysticks (left +, up +)
    state.LS_X = _apply_deadzone(-_stick(lx), deadzone)
    state.LS_Y = _apply_deadzone(_stick(ly), deadzone)
    state.RS_X = _apply_deadzone(-_stick(rx), deadzone)
    state.RS_Y = _apply_deadzone(_stick(ry), deadzone)

    # triggers (released = 0, pressed = 1)
    state.LT = report[4] / 255.0
    state.RT = report[5] / 255.0

    # D-PAD
    state.L_DPAD = float(bit("DLEFT"))
    state.R_DPAD = float(bit("DRIGHT"))
    state.U_DPAD = float(bit("DUP"))
    state.D_DPAD = float(bit("DDOWN"))

    # buttons
    state.A = bit("A")
    state.B = bit("B")
    state.X = bit("X")
    state.Y = bit("Y")
    state.LB = bit("LB")
    state.RB = bit("RB")
    state.LMB = bit("BACK")
    state.RMB = bit("START")
    state.XBOX = bit("XBOX")
    state.LS = bit("LS")
    state.RS = bit("RS")

    return state


class XInputUSBJoystick:
    """
    Background reader for an XInput gamepad over USB. Reconnects automatically
    (e.g. after the wireless dongle goes idle). Usage:
        js = XInputUSBJoystick(deadzone=0.05); js.start()
        state = js.get_state()   # JoystickState, or None if disconnected
        js.stop()
    """

    def __init__(self, deadzone: float = 0.05):
        self.deadzone = deadzone
        self._lock = threading.Lock()
        self._state = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_state(self):
        with self._lock:
            return self._state

    def _reader_loop(self):
        warned = None
        while not self._stop.is_set():
            found = _find_xinput_device()
            if found is None:
                if warned != "missing":
                    print("[xinput_usb] No XInput controller found (dongle idle or unplugged). Waiting...")
                    warned = "missing"
                time.sleep(1.0)
                continue
            sysdev, node = found
            try:
                pad = _XInputUSBDevice(node)
            except PermissionError:
                if warned != "perm":
                    print(f"[xinput_usb] No permission to open {node}. Add the udev rule or run with sudo.")
                    warned = "perm"
                time.sleep(1.0)
                continue
            except OSError as e:
                if warned != "busy":
                    print(f"[xinput_usb] Could not claim {node}: {e} (is a kernel driver bound to it?)")
                    warned = "busy"
                time.sleep(1.0)
                continue

            print(f"[xinput_usb] Controller opened: {os.path.basename(sysdev)} ({node})")
            warned = None
            # XInput only sends reports on change, so start from the neutral state
            # (the controller is awake: the dongle only enumerates with this ID when it is)
            with self._lock:
                self._state = JoystickState()
            try:
                while not self._stop.is_set():
                    report = pad.read_report()
                    if report is None:
                        continue
                    state = xinput_report_to_joystick_state(report, self.deadzone)
                    with self._lock:
                        self._state = state
            except OSError as e:
                print(f"[xinput_usb] Controller lost ({e}).")
            finally:
                with self._lock:
                    self._state = None
                pad.close()
