#!/usr/bin/env python3
import ctypes
import os
import subprocess
import uuid
from pathlib import Path

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_OPENGL_API = 0x30A2
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_ALPHA_SIZE = 0x3021
EGL_DEPTH_SIZE = 0x3025
EGL_NONE = 0x3038
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056

EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
EGL_DEVICE_UUID_EXT = 0x335C
EGL_DRIVER_UUID_EXT = 0x335D
EGL_DRIVER_NAME_EXT = 0x335E
EGL_RENDERER_EXT = 0x335F

GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02


class _SilenceStderr:
    def __enter__(self):
        self._old = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)

    def __exit__(self, *_exc):
        os.dup2(self._old, 2)
        os.close(self._devnull)
        os.close(self._old)


def _norm_pci_bus_id(bus_id: str) -> str:
    dom, rest = bus_id.split(":", 1)
    return f"{dom[-4:]}:{rest}".lower()


def _egl_proc(lib, name, restype, argtypes):
    addr = lib.eglGetProcAddress(name.encode())
    assert addr, f"Missing EGL proc: {name}"
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)


def _pci_slot_name_from_drm_node(drm_path: str) -> str | None:
    uevent = Path("/sys/class/drm") / Path(drm_path).name / "device" / "uevent"
    if not uevent.exists():
        return None
    for line in uevent.read_text().splitlines():
        if line.startswith("PCI_SLOT_NAME="):
            return line.split("=", 1)[1]
    return None


def list_egl_devices() -> list[tuple[int, ctypes.c_void_p, str | None, str | None]]:
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    EGLint = ctypes.c_int
    EGLBoolean = ctypes.c_uint
    EGLDeviceEXT = ctypes.c_void_p

    eglQueryDevicesEXT = _egl_proc(
        lib,
        "eglQueryDevicesEXT",
        EGLBoolean,
        [EGLint, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(EGLint)],
    )
    eglQueryDeviceStringEXT = _egl_proc(
        lib,
        "eglQueryDeviceStringEXT",
        ctypes.c_char_p,
        [EGLDeviceEXT, EGLint],
    )
    eglQueryDeviceBinaryEXT = _egl_proc(
        lib,
        "eglQueryDeviceBinaryEXT",
        EGLBoolean,
        [EGLDeviceEXT, EGLint, EGLint, ctypes.c_void_p, ctypes.POINTER(EGLint)],
    )

    n = EGLint()
    max_devices = 32
    devices = (EGLDeviceEXT * max_devices)()
    ok = eglQueryDevicesEXT(max_devices, devices, ctypes.byref(n))
    assert ok, "eglQueryDevicesEXT failed"

    out = []
    for i in range(n.value):
        dev = devices[i]
        drm = eglQueryDeviceStringEXT(dev, EGL_DRM_RENDER_NODE_FILE_EXT) or eglQueryDeviceStringEXT(
            dev, EGL_DRM_DEVICE_FILE_EXT
        )
        drm = drm.decode() if drm else None
        pci = _pci_slot_name_from_drm_node(drm) if drm else None
        dev_uuid = (ctypes.c_ubyte * 16)()
        size = EGLint()
        ok_uuid = eglQueryDeviceBinaryEXT(dev, EGL_DEVICE_UUID_EXT, 16, dev_uuid, ctypes.byref(size))
        dev_uuid = bytes(dev_uuid) if ok_uuid and size.value == 16 else None
        drv_uuid = (ctypes.c_ubyte * 16)()
        size = EGLint()
        ok_uuid = eglQueryDeviceBinaryEXT(dev, EGL_DRIVER_UUID_EXT, 16, drv_uuid, ctypes.byref(size))
        drv_uuid = bytes(drv_uuid) if ok_uuid and size.value == 16 else None
        drv_name = eglQueryDeviceStringEXT(dev, EGL_DRIVER_NAME_EXT)
        drv_name = drv_name.decode() if drv_name else None
        dev_name = eglQueryDeviceStringEXT(dev, EGL_RENDERER_EXT)
        dev_name = dev_name.decode() if dev_name else None
        out.append((i, dev, drm, pci, dev_uuid, drv_uuid, drv_name, dev_name))
    return out


def nvidia_smi_uuid_to_info() -> dict[bytes, tuple[int, str, str, str]]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
        text=True,
    )
    m: dict[bytes, tuple[int, str, str, str]] = {}
    for line in out.splitlines():
        idx, gpu_uuid, bus_id, name = [x.strip() for x in line.split(",", 3)]
        gpu_uuid = gpu_uuid.removeprefix("GPU-")
        m[uuid.UUID(gpu_uuid).bytes] = (int(idx), f"GPU-{gpu_uuid}", _norm_pci_bus_id(bus_id), name)
    return m


def probe_opengl_for_egl_device(dev: ctypes.c_void_p) -> tuple[str | None, str | None, str | None]:
    egl = ctypes.CDLL("libEGL.so.1")
    gl = ctypes.CDLL("libGL.so.1")

    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    egl.eglChooseConfig.restype = ctypes.c_uint
    egl.eglChooseConfig.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    egl.eglBindAPI.restype = ctypes.c_uint
    egl.eglBindAPI.argtypes = [ctypes.c_uint]
    egl.eglCreateContext.restype = ctypes.c_void_p
    egl.eglCreateContext.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
    egl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    egl.eglMakeCurrent.restype = ctypes.c_uint
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    egl.eglDestroySurface.restype = ctypes.c_uint
    egl.eglDestroySurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    egl.eglDestroyContext.restype = ctypes.c_uint
    egl.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    egl.eglTerminate.restype = ctypes.c_uint
    egl.eglTerminate.argtypes = [ctypes.c_void_p]

    eglGetPlatformDisplayEXT = _egl_proc(
        egl,
        "eglGetPlatformDisplayEXT",
        ctypes.c_void_p,
        [ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
    )

    with _SilenceStderr():
        dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, dev, None)
        if not dpy:
            return None, None, None
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not egl.eglInitialize(dpy, ctypes.byref(major), ctypes.byref(minor)):
            return None, None, None

        egl.eglBindAPI(EGL_OPENGL_API)
        attrs = (ctypes.c_int * 17)(
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_DEPTH_SIZE,
            24,
            EGL_NONE,
            EGL_NONE,
            EGL_NONE,
        )
        cfg = ctypes.c_void_p()
        ncfg = ctypes.c_int()
        if not egl.eglChooseConfig(dpy, attrs, ctypes.byref(cfg), 1, ctypes.byref(ncfg)) or ncfg.value < 1:
            egl.eglTerminate(dpy)
            return None, None, None

        ctx_attrs = (ctypes.c_int * 1)(EGL_NONE)
        ctx = egl.eglCreateContext(dpy, cfg, ctypes.c_void_p(0), ctx_attrs)
        pb_attrs = (ctypes.c_int * 5)(EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE)
        surf = egl.eglCreatePbufferSurface(dpy, cfg, pb_attrs)
        if not ctx or not surf or not egl.eglMakeCurrent(dpy, surf, surf, ctx):
            if surf:
                egl.eglDestroySurface(dpy, surf)
            if ctx:
                egl.eglDestroyContext(dpy, ctx)
            egl.eglTerminate(dpy)
            return None, None, None

        gl.glGetString.restype = ctypes.c_char_p
        gl.glGetString.argtypes = [ctypes.c_uint]
        vendor = (gl.glGetString(GL_VENDOR) or b"").decode(errors="ignore") or None
        renderer = (gl.glGetString(GL_RENDERER) or b"").decode(errors="ignore") or None
        version = (gl.glGetString(GL_VERSION) or b"").decode(errors="ignore") or None

        egl.eglMakeCurrent(dpy, ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_void_p(0))
        egl.eglDestroySurface(dpy, surf)
        egl.eglDestroyContext(dpy, ctx)
        egl.eglTerminate(dpy)
        return vendor, renderer, version


def main():
    keys = ["MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]
    print("env:", " ".join(f"{k}={os.environ.get(k)}" for k in keys))
    smi = nvidia_smi_uuid_to_info()
    for egl_id, dev, drm, pci, dev_uuid, drv_uuid, drv_name, dev_name in list_egl_devices():
        vendor, renderer, version = probe_opengl_for_egl_device(dev)
        smi_info = smi.get(dev_uuid) if dev_uuid else None
        smi_idx = smi_info[0] if smi_info else None
        smi_uuid = smi_info[1] if smi_info else None
        smi_pci = smi_info[2] if smi_info else None
        smi_name = smi_info[3] if smi_info else None
        dev_uuid_s = str(uuid.UUID(bytes=dev_uuid)) if dev_uuid else None
        drv_uuid_s = str(uuid.UUID(bytes=drv_uuid)) if drv_uuid else None
        print(
            f"MUJOCO_EGL_DEVICE_ID={egl_id} nvidia-smi={smi_idx} smi_uuid={smi_uuid} smi_pci={smi_pci} smi_name={smi_name} "
            f"egl_dev_uuid={dev_uuid_s} egl_drv_uuid={drv_uuid_s} egl_driver={drv_name} egl_name={dev_name} "
            f"drm={drm} pci={pci} gl_vendor={vendor} gl_renderer={renderer} gl_version={version}"
        )


if __name__ == "__main__":
    main()
