"""系统环境工具：只读采集本机软硬件、网络、运行时与进程信息。

Windows 通过 CIM / PowerShell 采集，Linux / macOS 回退到 /proc、sysctl、ip、df 等命令。
输出按节组织并有长度上限，环境变量中的敏感值统一脱敏。
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import getpass
import json
import locale
import os
import platform
import re
import shutil
import socket
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from .context import ToolContext
from .env_check import _SENSITIVE_KEYS, _mask_value
from .registry import RISK_LOW, tool_spec

_MAX_SECTION_CHARS = 3200
_MAX_TOTAL_CHARS = 10000

_SECTION_ORDER = [
    "overview", "os", "cpu", "memory", "disk", "gpu",
    "network", "runtimes", "packages", "software", "processes", "battery", "env",
]
_LABELS = {
    "overview": "概览",
    "os": "操作系统",
    "cpu": "处理器",
    "memory": "内存",
    "disk": "磁盘",
    "gpu": "显卡",
    "network": "网络",
    "runtimes": "运行时与开发工具",
    "packages": "项目与全局依赖",
    "software": "已安装软件",
    "processes": "进程（按内存排序）",
    "battery": "电池",
    "env": "环境变量",
}
_ALIASES = {
    "system": "os",
    "platform": "os",
    "hardware": "cpu,memory,disk,gpu",
    "hw": "cpu,memory,disk,gpu",
    "net": "network",
    "network": "network",
    "runtime": "runtimes",
    "tools": "runtimes",
    "package": "packages",
    "packages": "packages",
    "pkg": "packages",
    "deps": "packages",
    "dependencies": "packages",
    "project": "packages",
    "projects": "packages",
    "apps": "software",
    "app": "software",
    "process": "processes",
    "proc": "processes",
    "environment": "env",
    "envs": "env",
}


def _is_windows() -> bool:
    return sys.platform == "win32"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


async def _run(args: list[str], timeout: float = 8.0) -> str:
    """执行只读命令并合并 stderr；超时或命令缺失时返回空串。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (FileNotFoundError, OSError):
        return ""
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return ""
    return _decode(out).strip()


def _is_windows_script(path: str) -> bool:
    return path.lower().endswith((".cmd", ".bat"))


def _exe_args(exe: str, args: list[str]) -> list[str]:
    # npm/pnpm/code 等在 Windows 上是 .cmd，必须经 cmd /c 才能执行
    if _is_windows() and _is_windows_script(exe):
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c", exe, *args]
    return [exe, *args]


def _powershell() -> str:
    return shutil.which("powershell") or "powershell"


def _json_load(text: str):
    text = (text or "").strip().lstrip("\ufeff")
    if not text:
        return None
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        text = text[min(starts):]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


async def _ps_json(script: str, timeout: float = 20.0):
    wrapped = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$WarningPreference='SilentlyContinue';"
        "$ProgressPreference='SilentlyContinue';"
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        + script
    )
    return _json_load(await _run(
        [_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", wrapped],
        timeout,
    ))


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _human_bytes(num) -> str:
    try:
        value = float(num)
    except (TypeError, ValueError):
        return "未知"
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.0f} {units[idx]}" if idx == 0 else f"{value:.1f} {units[idx]}"


def _wmi_time(value) -> str:
    text = str(value or "").strip()
    match = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", text)
    if match:
        return f"{match[1]}-{match[2]}-{match[3]} {match[4]}:{match[5]}:{match[6]}"
    match = re.search(r"/Date\((\d+)\)/", text)
    if match:
        return datetime.fromtimestamp(int(match[1]) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    return text or "未知"


def _is_admin() -> bool:
    try:
        if _is_windows():
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except (AttributeError, OSError):
        return False


def _uptime_seconds() -> float | None:
    try:
        if _is_windows():
            return ctypes.windll.kernel32.GetTickCount64() / 1000
        if sys.platform.startswith("linux"):
            return float(Path("/proc/uptime").read_text().split()[0])
    except (AttributeError, OSError, ValueError):
        return None
    return None


def _duration_text(seconds: float) -> str:
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours or days:
        parts.append(f"{hours} 小时")
    parts.append(f"{minutes} 分钟")
    return " ".join(parts)


def _tz_text() -> str:
    try:
        is_dst = time.localtime().tm_isdst > 0 and time.daylight
        name = time.tzname[1 if is_dst else 0] or "未知时区"
        offset = -(time.altzone if is_dst else time.timezone)
        sign = "+" if offset >= 0 else "-"
        hours, minutes = divmod(abs(offset) // 60, 60)
        return f"{name} (UTC{sign}{hours:02d}:{minutes:02d})"
    except (IndexError, OSError, ValueError):
        return "未知"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _win_memory() -> dict | None:
    try:
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return {
            "total": stat.ullTotalPhys,
            "avail": stat.ullAvailPhys,
            "load": stat.dwMemoryLoad,
            "page_total": stat.ullTotalPageFile,
            "page_avail": stat.ullAvailPageFile,
        }
    except (AttributeError, OSError):
        return None


def _linux_meminfo() -> dict[str, int] | None:
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(errors="replace").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                info[key] = int(parts[0]) * 1024
        if "MemTotal" not in info:
            return None
        total = info["MemTotal"]
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        info["avail"] = avail
        info["load"] = round((total - avail) * 100 / total) if total else 0
        return info
    except OSError:
        return None


async def _memory_totals() -> dict | None:
    if _is_windows():
        return _win_memory()
    info = _linux_meminfo()
    if info:
        return info
    if sys.platform == "darwin":
        code, total = await _run(["sysctl", "-n", "hw.memsize"])
        if code and total.isdigit():
            total_bytes = int(total)
            free_bytes = 0
            _, vm = await _run(["vm_stat"])
            match = re.search(r"Pages free:\s+(\d+)", vm)
            if match:
                free_bytes = int(match.group(1)) * 4096
            return {
                "total": total_bytes,
                "avail": free_bytes,
                "load": round((total_bytes - free_bytes) * 100 / total_bytes) if total_bytes else 0,
            }
    return None


def _overview() -> str:
    lines = ["【概览】"]
    admin = "，管理员" if _is_admin() else ""
    lines.append(f"- 计算机: {socket.gethostname()}（用户 {getpass.getuser()}{admin}）")
    os_text = f"{platform.system()} {platform.release()} {platform.version()}".strip()
    lines.append(f"- 系统: {os_text}（{platform.machine() or '未知架构'}）")
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "未知型号"
    lines.append(f"- CPU: {processor}（{os.cpu_count() or '?'} 逻辑核）")
    root = f"{os.environ.get('SYSTEMDRIVE', 'C:')}\\" if _is_windows() else "/"
    try:
        usage = shutil.disk_usage(root)
        lines.append(f"- 系统盘 {root}: 剩余 {_human_bytes(usage.free)} / 共 {_human_bytes(usage.total)}")
    except OSError:
        pass
    lines.append(f"- Python: {platform.python_version()}（{sys.executable}）")
    shell = os.environ.get("COMSPEC") if _is_windows() else os.environ.get("SHELL")
    if shell:
        lines.append(f"- Shell: {shell}")
    uptime = _uptime_seconds()
    if uptime:
        lines.append(f"- 开机时长: {_duration_text(uptime)}")
    lines.append(f"- 时区: {_tz_text()}")
    lines.append(f"- 采集时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return "\n".join(lines)


async def _section_overview(_keyword: str, _limit: int, _workspace: str = "") -> str:
    text = _overview()
    mem = await _memory_totals()
    if mem:
        usage_line = (
            f"- 内存: 可用 {_human_bytes(mem['avail'])} / 共 {_human_bytes(mem['total'])}"
            f"（已用 {mem.get('load', '?')}%）"
        )
        text = text.replace("- 系统盘", f"{usage_line}\n- 系统盘", 1)
    return text


async def _section_os(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【操作系统】"]
    if _is_windows():
        data = await _ps_json(
            "$os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,"
            "OSArchitecture,InstallDate,LastBootUpTime,SystemDrive,WindowsDirectory;"
            "$cs=Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,Domain,Workgroup;"
            "$bios=Get-CimInstance Win32_BIOS | Select-Object Manufacturer,SMBIOSBIOSVersion,ReleaseDate;"
            "[pscustomobject]@{os=$os;cs=$cs;bios=$bios} | ConvertTo-Json -Depth 4 -Compress",
            timeout=25,
        ) or {}
        os_info = data.get("os") or {}
        cs = data.get("cs") or {}
        bios = data.get("bios") or {}
        if os_info.get("Caption"):
            lines.append(
                f"- 版本: {os_info['Caption']} {os_info.get('Version', '')}"
                f"（Build {os_info.get('BuildNumber', '?')}，{os_info.get('OSArchitecture', '未知架构')}）"
            )
        vendor_model = " ".join(
            str(cs.get(key)).strip() for key in ("Manufacturer", "Model") if cs.get(key)
        )
        if vendor_model:
            lines.append(f"- 设备: {vendor_model}")
        if bios:
            lines.append(
                f"- BIOS: {bios.get('Manufacturer', '')} {bios.get('SMBIOSBIOSVersion', '')}"
                f"（{_wmi_time(bios.get('ReleaseDate'))}）"
            )
        domain = cs.get("Domain") or cs.get("Workgroup")
        if domain:
            lines.append(f"- 域/工作组: {domain}")
        if os_info.get("InstallDate"):
            lines.append(f"- 安装时间: {_wmi_time(os_info['InstallDate'])}")
        if os_info.get("LastBootUpTime"):
            lines.append(f"- 上次启动: {_wmi_time(os_info['LastBootUpTime'])}")
        if os_info.get("WindowsDirectory"):
            lines.append(f"- 系统目录: {os_info['WindowsDirectory']}")
    else:
        uname = platform.uname()
        lines.append(f"- 内核: {uname.system} {uname.release} {uname.version}")
        lines.append(f"- 主机名: {uname.node}（{uname.machine}）")
        if sys.platform == "darwin":
            _, sw = await _run(["sw_vers"])
            for line in sw.splitlines():
                lines.append(f"- {line.strip()}")
        elif Path("/etc/os-release").exists():
            for line in Path("/etc/os-release").read_text(errors="replace").splitlines():
                if line.startswith(("PRETTY_NAME=", "VERSION=")):
                    key, _, value = line.partition("=")
                    lines.append(f"- {key.title()}: {value.strip().strip(chr(34))}")
    uptime = _uptime_seconds()
    if uptime:
        lines.append(f"- 开机时长: {_duration_text(uptime)}")
    lines.append(f"- 当前用户: {getpass.getuser()}（{'管理员' if _is_admin() else '普通用户'}）")
    lines.append(f"- 时区: {_tz_text()}")
    try:
        language, _ = locale.getlocale()
        lines.append(f"- 语言/编码: {language or '未知'}/{locale.getencoding()}")
    except (ValueError, TypeError):
        pass
    lines.append(f"- Python: {platform.python_version()}（{sys.executable}）")
    return "\n".join(lines)


async def _section_cpu(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【处理器】"]
    if _is_windows():
        data = _as_list(await _ps_json(
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
            "NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed,LoadPercentage,"
            "L2CacheSize,L3CacheSize,AddressWidth | ConvertTo-Json -Depth 3 -Compress",
            timeout=20,
        ))
        for cpu in data[:4]:
            name = str(cpu.get("Name") or "未知型号").strip()
            lines.append(f"- {name}")
            lines.append(
                f"  - 核心/线程: {cpu.get('NumberOfCores', '?')}/{cpu.get('NumberOfLogicalProcessors', '?')}"
                f"；主频: {cpu.get('MaxClockSpeed', '?')} MHz；当前负载: {cpu.get('LoadPercentage', '?')}%"
            )
            cache = [f"L{level}: {cpu[key]} KB" for level, key in ((2, "L2CacheSize"), (3, "L3CacheSize")) if cpu.get(key)]
            if cache:
                lines.append(f"  - 缓存: {'，'.join(cache)}；位宽: {cpu.get('AddressWidth', '?')} 位")
        if not data:
            lines.append(f"- {platform.processor() or '未知型号'}（{os.cpu_count() or '?'} 逻辑核）")
        return "\n".join(lines)
    if sys.platform.startswith("linux"):
        try:
            info = Path("/proc/cpuinfo").read_text(errors="replace")
        except OSError:
            info = ""
        models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", info, re.M)))
        lines.append(f"- 型号: {models[0].strip() if models else platform.processor() or '未知型号'}")
        lines.append(f"- 逻辑核: {os.cpu_count() or '?'}")
        freq = re.search(r"^cpu MHz\s*:\s*([\d.]+)", info, re.M)
        if freq:
            lines.append(f"- 当前主频: {float(freq.group(1)):.0f} MHz")
        try:
            load = Path("/proc/loadavg").read_text().split()[:3]
            lines.append(f"- 平均负载（1/5/15 分钟）: {' / '.join(load)}")
        except OSError:
            pass
        return "\n".join(lines)
    if sys.platform == "darwin":
        _, brand = await _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        _, physical = await _run(["sysctl", "-n", "hw.physicalcpu"])
        _, logical = await _run(["sysctl", "-n", "hw.logicalcpu"])
        _, load = await _run(["sysctl", "-n", "vm.loadavg"])
        lines.append(f"- 型号: {brand or platform.processor() or '未知型号'}")
        lines.append(f"- 核心/线程: {physical or '?'}/{logical or os.cpu_count() or '?'}")
        if load:
            lines.append(f"- 平均负载: {load.strip('{} ')}")
        return "\n".join(lines)
    lines.append(f"- {platform.processor() or '未知型号'}（{os.cpu_count() or '?'} 逻辑核）")
    return "\n".join(lines)


async def _section_memory(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【内存】"]
    mem = await _memory_totals()
    if not mem:
        lines.append("- 未能读取内存信息，可改用 terminal 执行系统自带的内存查询命令。")
        return "\n".join(lines)
    lines.append(f"- 物理内存: 共 {_human_bytes(mem['total'])}，可用 {_human_bytes(mem['avail'])}，已用 {mem.get('load', '?')}%")
    if _is_windows():
        if mem.get("page_total"):
            lines.append(
                f"- 提交内存/页面文件: 共 {_human_bytes(mem['page_total'])}，可用 {_human_bytes(mem['page_avail'])}"
            )
    elif sys.platform.startswith("linux") and "SwapTotal" in mem:
        lines.append(
            f"- 交换分区: 共 {_human_bytes(mem['SwapTotal'])}，可用 {_human_bytes(mem.get('SwapFree', 0))}"
        )
        if mem.get("Buffers") or mem.get("Cached"):
            lines.append(
                f"- 缓存: Buffers {_human_bytes(mem.get('Buffers', 0))}，Cached {_human_bytes(mem.get('Cached', 0))}"
            )
    elif sys.platform == "darwin":
        _, swap = await _run(["sysctl", "-n", "vm.swapusage"])
        if swap:
            lines.append(f"- 交换分区: {swap.strip()}")
    return "\n".join(lines)


async def _section_disk(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【磁盘】"]
    if _is_windows():
        data = await _ps_json(
            "$volumes=Get-CimInstance Win32_LogicalDisk -Filter \"DriveType=3\" | "
            "Select-Object DeviceID,VolumeName,FileSystem,Size,FreeSpace;"
            "$disks=Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType,MediaType,Partitions;"
            "[pscustomobject]@{volumes=$volumes;disks=$disks} | ConvertTo-Json -Depth 3 -Compress",
            timeout=20,
        ) or {}
        for vol in _as_list(data.get("volumes")):
            size = vol.get("Size") or 0
            free = vol.get("FreeSpace") or 0
            used_pct = round((size - free) * 100 / size) if size else 0
            name = vol.get("VolumeName") or ""
            label = f"{vol.get('DeviceID', '?')}" + (f"（{name}）" if name else "")
            lines.append(
                f"- {label} {vol.get('FileSystem', '')}: 共 {_human_bytes(size)}，"
                f"剩余 {_human_bytes(free)}（已用 {used_pct}%）"
            )
        for disk in _as_list(data.get("disks"))[:6]:
            parts = [
                str(disk.get("Model") or "未知型号").strip(),
                _human_bytes(disk.get("Size")),
            ]
            if disk.get("InterfaceType"):
                parts.append(str(disk["InterfaceType"]))
            if disk.get("MediaType"):
                parts.append(str(disk["MediaType"]))
            lines.append(f"- 物理磁盘: {'，'.join(parts)}")
        if len(lines) == 1:
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                root = f"{letter}:\\"
                if not os.path.exists(root):
                    continue
                with contextlib.suppress(OSError):
                    usage = shutil.disk_usage(root)
                    lines.append(
                        f"- {root}: 共 {_human_bytes(usage.total)}，剩余 {_human_bytes(usage.free)}"
                    )
        return "\n".join(lines)
    _, df = await _run(["df", "-kP"], timeout=10)
    rows = []
    for line in df.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        fs, blocks, used, avail, _capacity, mount = parts[0], parts[1], parts[2], parts[3], parts[4], " ".join(parts[5:])
        if fs.startswith("/dev/loop") or fs in {"devfs", "udev"} or mount.startswith("/System/Volumes/"):
            continue
        if not blocks.isdigit():
            continue
        rows.append((mount, fs, int(blocks) * 1024, int(avail) * 1024, int(used) * 1024))
    for mount, fs, total, avail, used in rows[:10]:
        pct = round(used * 100 / total) if total else 0
        lines.append(
            f"- {mount}（{fs}）: 共 {_human_bytes(total)}，剩余 {_human_bytes(avail)}（已用 {pct}%）"
        )
    if len(lines) == 1:
        lines.append("- 未能读取磁盘分区信息。")
    return "\n".join(lines)


async def _section_gpu(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【显卡】"]
    smi = shutil.which("nvidia-smi")
    if smi:
        out = await _run(
            _exe_args(smi, ["--query-gpu=name,memory.total,driver_version,temperature.gpu,utilization.gpu", "--format=csv,noheader"]),
            timeout=10,
        )
        for line in out.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 3:
                extra = []
                if len(fields) > 3 and fields[3]:
                    extra.append(f"温度 {fields[3]}°C")
                if len(fields) > 4 and fields[4]:
                    extra.append(f"占用 {fields[4].replace(' ', '')}")
                suffix = f"；{'，'.join(extra)}" if extra else ""
                lines.append(f"- {fields[0]}（显存 {fields[1]}，驱动 {fields[2]}{suffix}）")
    if _is_windows():
        data = _as_list(await _ps_json(
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,"
            "VideoProcessor,CurrentHorizontalResolution,CurrentVerticalResolution,Status | "
            "ConvertTo-Json -Depth 3 -Compress",
            timeout=20,
        ))
        seen = {line.split("（")[0].removeprefix("- ") for line in lines[1:]}
        for gpu in data:
            name = str(gpu.get("Name") or "未知显卡").strip()
            if name in seen:
                continue
            detail = []
            if gpu.get("AdapterRAM"):
                detail.append(f"显存 ≈ {_human_bytes(gpu['AdapterRAM'])}")
            if gpu.get("DriverVersion"):
                detail.append(f"驱动 {gpu['DriverVersion']}")
            if gpu.get("CurrentHorizontalResolution") and gpu.get("CurrentVerticalResolution"):
                detail.append(f"分辨率 {gpu['CurrentHorizontalResolution']}×{gpu['CurrentVerticalResolution']}")
            if gpu.get("Status"):
                detail.append(str(gpu["Status"]))
            lines.append(f"- {name}" + (f"（{'，'.join(detail)}）" if detail else ""))
    elif sys.platform.startswith("linux"):
        lspci = shutil.which("lspci")
        if lspci:
            out = await _run(_exe_args(lspci, []), timeout=8)
            for line in out.splitlines():
                if re.search(r"vga|3d|display", line, re.I):
                    lines.append(f"- {line.split(': ', 1)[-1]}")
    elif sys.platform == "darwin":
        out = await _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
        for line in out.splitlines():
            text = line.strip()
            if text.startswith(("Chipset Model:", "VRAM (Total):", "Metal Support:")):
                lines.append(f"- {text}")
    if len(lines) == 1:
        lines.append("- 未检测到可用显卡信息。")
    return "\n".join(lines)


async def _section_network(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【网络】"]
    lines.append(f"- 主机名: {socket.gethostname()}")
    if _is_windows():
        data = _as_list(await _ps_json(
            "Get-CimInstance Win32_NetworkAdapterConfiguration -Filter \"IPEnabled=TRUE\" | "
            "Select-Object Description,MACAddress,IPAddress,IPSubnet,DefaultIPGateway,DNSServerSearchOrder,DHCPEnabled | "
            "ConvertTo-Json -Depth 4 -Compress",
            timeout=20,
        ))
        for nic in data[:8]:
            name = str(nic.get("Description") or "未知网卡").strip()
            ips = _as_list(nic.get("IPAddress"))
            masks = _as_list(nic.get("IPSubnet"))
            pairs = [f"{ip}/{masks[i]}" if i < len(masks) and masks[i] else ip for i, ip in enumerate(ips)]
            detail = []
            if pairs:
                detail.append("IP " + "，".join(str(p) for p in pairs[:4]))
            gateways = _as_list(nic.get("DefaultIPGateway"))
            if gateways:
                detail.append("网关 " + "，".join(str(g) for g in gateways[:2]))
            dns = _as_list(nic.get("DNSServerSearchOrder"))
            if dns:
                detail.append("DNS " + "，".join(str(d) for d in dns[:3]))
            if nic.get("MACAddress"):
                detail.append(f"MAC {nic['MACAddress']}")
            if nic.get("DHCPEnabled") is not None:
                detail.append(f"DHCP {'是' if nic['DHCPEnabled'] else '否'}")
            lines.append(f"- {name}" + (f": {'；'.join(detail)}" if detail else ""))
        if len(lines) == 2:
            lines.append("- 未获取到已启用的网络适配器信息。")
        return "\n".join(lines)
    ip_cmd = shutil.which("ip")
    if ip_cmd:
        out = await _run(_exe_args(ip_cmd, ["-brief", "address"]), timeout=8)
        for line in out.splitlines()[:10]:
            lines.append(f"- {line.strip()}")
        route = await _run(_exe_args(ip_cmd, ["route", "show", "default"]), timeout=8)
        if route:
            lines.append(f"- 默认路由: {route.splitlines()[0].strip()}")
    else:
        out = await _run(["ifconfig"], timeout=8)
        for line in out.splitlines():
            if re.match(r"^\w+\d*:?\s", line) or "inet " in line:
                lines.append(f"- {line.strip()}")
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        servers = re.findall(r"^nameserver\s+(\S+)", resolv.read_text(errors="replace"), re.M)
        if servers:
            lines.append(f"- DNS: {', '.join(servers[:3])}")
    return "\n".join(lines)


async def _runtime_versions() -> list[tuple[str, str]]:
    specs: list[tuple[str, str | None, list[str]]] = [
        ("Python", None, ["--version"]),
        ("pip", None, ["-m", "pip", "--version"]),
        ("Node.js", "node", ["--version"]),
        ("npm", "npm", ["--version"]),
        ("pnpm", "pnpm", ["--version"]),
        ("Yarn", "yarn", ["--version"]),
        ("Bun", "bun", ["--version"]),
        ("Deno", "deno", ["--version"]),
        ("npx", "npx", ["--version"]),
        ("Corepack", "corepack", ["--version"]),
        ("TypeScript", "tsc", ["--version"]),
        ("Vite", "vite", ["--version"]),
        ("vue-tsc", "vue-tsc", ["--version"]),
        ("ESLint", "eslint", ["--version"]),
        ("Prettier", "prettier", ["--version"]),
        ("Webpack", "webpack", ["--version"]),
        ("Vitest", "vitest", ["--version"]),
        ("Jest", "jest", ["--version"]),
        ("Playwright", "playwright", ["--version"]),
        ("Next.js", "next", ["--version"]),
        ("Nuxt", "nuxt", ["--version"]),
        ("Expo", "expo", ["--version"]),
        ("Git", "git", ["--version"]),
        ("Docker", "docker", ["--version"]),
        ("Podman", "podman", ["--version"]),
        ("uv", "uv", ["--version"]),
        ("Poetry", "poetry", ["--version"]),
        ("conda", "conda", ["--version"]),
        ("Java", "java", ["-version"]),
        ("Maven", "mvn", ["-version"]),
        ("Gradle", "gradle", ["--version"]),
        ("Go", "go", ["version"]),
        ("Rust", "rustc", ["--version"]),
        ("Cargo", "cargo", ["--version"]),
        (".NET", "dotnet", ["--version"]),
        ("GCC", "gcc", ["--version"]),
        ("Clang", "clang", ["--version"]),
        ("CMake", "cmake", ["--version"]),
        ("Make", "make", ["--version"]),
        ("VS Code", "code", ["--version"]),
        ("PowerShell", "powershell", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.ToString()"]),
        ("pwsh", "pwsh", ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable.PSVersion.ToString()"]),
        ("psql", "psql", ["--version"]),
        ("MySQL", "mysql", ["--version"]),
        ("Redis CLI", "redis-cli", ["--version"]),
        ("SQLite", "sqlite3", ["--version"]),
        ("Ninja", "ninja", ["--version"]),
        ("Meson", "meson", ["--version"]),
        ("Bazel", "bazel", ["--version"]),
        ("PHP", "php", ["--version"]),
        ("Composer", "composer", ["--version"]),
        ("Ruby", "ruby", ["--version"]),
        ("Gem", "gem", ["--version"]),
        ("Dart", "dart", ["--version"]),
        ("Kotlin", "kotlinc", ["-version"]),
        ("Julia", "julia", ["--version"]),
        ("R", "Rscript", ["--version"]),
        ("Lua", "lua", ["-v"]),
        ("Swift", "swift", ["--version"]),
        ("kubectl", "kubectl", ["version", "--client"]),
    ]
    # 错误输出（缺模块、损坏的启动器）不能当作版本号展示
    bad_output = re.compile(
        r"no module named|could not be loaded|not recognized|command not found|"
        r"不是内部或外部命令|无法将|系统找不到|no such file",
        re.IGNORECASE,
    )
    jobs: list[tuple[str, list[str]]] = []
    for label, exe_name, args in specs:
        if exe_name is None:
            if label == "pip" and (pip_exe := shutil.which("pip")):
                jobs.append((label, _exe_args(pip_exe, ["--version"])))
            else:
                jobs.append((label, [sys.executable, *args]))
            continue
        exe = shutil.which(exe_name)
        if exe:
            jobs.append((label, _exe_args(exe, args)))
    results = await asyncio.gather(*[_run(cmd, timeout=8) for _, cmd in jobs])
    found = []
    for (label, _), output in zip(jobs, results, strict=True):
        first = next((line.strip() for line in output.splitlines() if line.strip()), "")
        if not first or bad_output.search(first):
            continue
        if label == "Python":
            first = first.removeprefix("Python ").strip()
        found.append((label, first[:160]))
    return found


async def _section_runtimes(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【运行时与开发工具】"]
    found = await _runtime_versions()
    if not found:
        lines.append("- 未在 PATH 中检测到常见运行时与开发工具。")
        return "\n".join(lines)
    python_line = ""
    for label, version in found:
        if label == "Python":
            python_line = f"- Python: {version}（{sys.executable}）"
            continue
        lines.append(f"- {label}: {version}")
    if python_line:
        lines.insert(1, python_line)
    lines.append("- 仅列出 PATH 中可检测到的工具；未列出的视为未安装或未加入 PATH。")
    return "\n".join(lines)


_MANIFEST_NAMES = ("package.json", "pyproject.toml", "requirements.txt", "Cargo.toml", "go.mod")
_SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "dist", "build", ".next", "out",
    "target", "__pycache__", ".idea", ".vs", ".cache", "appdata", "library",
    "windows", "program files", "program files (x86)", "$recycle.bin",
}


def _read_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _find_manifests(root: Path, max_depth: int = 2, limit: int = 8) -> list[Path]:
    """收集工作区及浅层子目录中的依赖清单，跳过 node_modules 等重型目录。"""
    found: list[Path] = []
    visited = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal visited
        if len(found) >= limit or visited >= 300 or depth > max_depth:
            return
        visited += 1
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and entry.name in _MANIFEST_NAMES:
                found.append(entry)
                if len(found) >= limit:
                    return
        if depth == max_depth:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name.casefold() in _SKIP_DIRS:
                continue
            walk(entry, depth + 1)

    if root.is_dir():
        walk(root, 0)
    return found


def _node_pm(directory: Path) -> str:
    for lock, name in (
        ("bun.lockb", "bun"), ("bun.lock", "bun"), ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"), ("package-lock.json", "npm"),
    ):
        if (directory / lock).exists():
            return name
    meta = _read_json_file(directory / "package.json") or {}
    declared = str(meta.get("packageManager") or "").split("@")[0]
    return declared or "npm"


def _python_pm(directory: Path) -> str:
    for lock, name in (("uv.lock", "uv"), ("poetry.lock", "poetry"), ("pdm.lock", "pdm")):
        if (directory / lock).exists():
            return name
    return "pip"


def _node_dep_rows(directory: Path, keyword: str, limit: int) -> tuple[list[str], int, int, int]:
    data = _read_json_file(directory / "package.json") or {}
    deps: dict[str, str] = {}
    for key in ("dependencies", "devDependencies"):
        for name, spec in (data.get(key) or {}).items():
            deps.setdefault(name, str(spec))
    installed = 0
    matched: list[str] = []
    for name in sorted(deps, key=str.casefold):
        meta = _read_json_file(directory / "node_modules" / name / "package.json") or {}
        version = str(meta.get("version") or "")
        if version:
            installed += 1
        if keyword and keyword.casefold() not in name.casefold():
            continue
        spec = deps[name]
        matched.append(f"{name}: 声明 {spec}" + (f"，已安装 {version}" if version else "（未安装）"))
    return matched[:limit], installed, len(deps), len(matched)


def _python_dep_rows(path: Path, keyword: str, limit: int) -> tuple[list[str], int]:
    names: list[str] = []
    if path.name == "requirements.txt":
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            item = line.strip()
            if item and not item.startswith(("#", "-")):
                names.append(item)
    else:
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError):
            data = {}
        project = data.get("project") or {}
        names.extend(str(dep) for dep in (project.get("dependencies") or []))
        for group in (project.get("optional-dependencies") or {}).values():
            names.extend(str(dep) for dep in (group or []))
    matched = [name for name in names if not keyword or keyword.casefold() in name.casefold()]
    return matched[:limit], len(matched)


def _cargo_dep_rows(path: Path, keyword: str, limit: int) -> tuple[list[str], int]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError):
        data = {}
    deps = data.get("dependencies") or {}
    matched = []
    for name, spec in sorted(deps.items(), key=lambda item: item[0].casefold()):
        if keyword and keyword.casefold() not in name.casefold():
            continue
        if isinstance(spec, dict):
            spec = spec.get("version") or ""
        matched.append(f"{name}: {spec}".rstrip(": "))
    return matched[:limit], len(matched)


def _go_dep_rows(path: Path, keyword: str, limit: int) -> tuple[list[str], int]:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        lines = []
    names: list[str] = []
    in_block = False
    for line in lines:
        text = line.strip()
        if text.startswith("require ("):
            in_block = True
            continue
        if in_block and text == ")":
            in_block = False
            continue
        if not in_block and not text.startswith("require "):
            continue
        parts = text.removeprefix("require ").split()
        if len(parts) >= 2 and parts[0] != "(":
            names.append(f"{parts[0]} {parts[1]}")
    matched = [name for name in names if not keyword or keyword.casefold() in name.casefold()]
    return matched[:limit], len(matched)


async def _global_npm_packages() -> tuple[str, list[str]]:
    """用 npm root -g 定位全局目录后直接读 package.json，比 npm ls -g 快一个数量级。"""
    npm = shutil.which("npm")
    if not npm:
        return "", []
    out = await _run(_exe_args(npm, ["root", "-g"]), timeout=15)
    root_text = next((line.strip() for line in out.splitlines() if line.strip()), "")
    root = Path(root_text) if root_text else None
    if not root or not root.is_dir():
        return "", []
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.casefold())
    except OSError:
        return "", []
    items: list[str] = []
    for entry in entries:
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        packages = []
        if entry.name.startswith("@"):
            try:
                packages = [sub for sub in sorted(entry.iterdir(), key=lambda p: p.name.casefold()) if sub.is_dir()]
            except OSError:
                continue
        else:
            packages = [entry]
        for folder in packages:
            meta = _read_json_file(folder / "package.json") or {}
            if meta.get("version"):
                items.append(f"{meta.get('name') or folder.name}: {meta['version']}")
    return str(root), items


async def _section_packages(keyword: str, limit: int, workspace: str = "") -> str:
    lines = ["【项目与全局依赖】"]
    root_text, global_items = await _global_npm_packages()
    if global_items:
        matched = [item for item in global_items if not keyword or keyword.casefold() in item.casefold()]
        lines.append(f"- 全局 npm 包（{root_text}）: 共 {len(global_items)} 个" + (f"，匹配 {len(matched)} 个" if keyword else ""))
        lines.extend(f"  - {item}" for item in matched[:limit])
        if len(matched) > limit:
            lines.append(f"  - …其余 {len(matched) - limit} 个未显示")

    root = Path(workspace or os.getcwd())
    manifests = _find_manifests(root)
    if not manifests:
        lines.append(f"- 工作区 {root} 未发现 package.json / pyproject.toml / requirements.txt / Cargo.toml / go.mod。")
        if len(lines) == 1:
            lines.append("- 未检测到全局 npm 包或项目依赖清单。")
        return "\n".join(lines)

    for path in manifests:
        folder = path.parent
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        if path.name == "package.json":
            rows, installed, total, matched = _node_dep_rows(folder, keyword, limit)
            lines.append(f"- {rel}（Node 项目，{_node_pm(folder)}；共 {total} 个依赖，已安装 {installed} 个）:")
        elif path.name in {"pyproject.toml", "requirements.txt"}:
            rows, matched = _python_dep_rows(path, keyword, limit)
            lines.append(f"- {rel}（Python 项目，{_python_pm(folder)}；{'匹配' if keyword else '共'} {matched} 项）:")
        elif path.name == "Cargo.toml":
            rows, matched = _cargo_dep_rows(path, keyword, limit)
            lines.append(f"- {rel}（Rust 项目，cargo；{'匹配' if keyword else '共'} {matched} 项）:")
        else:
            rows, matched = _go_dep_rows(path, keyword, limit)
            lines.append(f"- {rel}（Go 项目；{'匹配' if keyword else '共'} {matched} 项）:")
        if not rows:
            lines.append("  - 无名称匹配的依赖" if keyword else "  - 未解析到依赖声明")
        lines.extend(f"  - {row}" for row in rows)
        if matched > limit:
            lines.append(f"  - …其余 {matched - limit} 项未显示，可增大 limit 或使用 keyword")
    return "\n".join(lines)


async def _software_windows() -> list[tuple[str, str, str]]:
    data = _as_list(await _ps_json(
        "$keys=@('HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',"
        "'HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',"
        "'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*');"
        "Get-ItemProperty $keys -ErrorAction SilentlyContinue | "
        "Where-Object { $_.DisplayName -and $_.SystemComponent -ne 1 } | "
        "Select-Object DisplayName,DisplayVersion,Publisher | "
        "Sort-Object DisplayName | ConvertTo-Json -Depth 3 -Compress",
        timeout=30,
    ))
    items = []
    for row in data:
        name = str(row.get("DisplayName") or "").strip()
        if name:
            items.append((
                name,
                str(row.get("DisplayVersion") or "").strip(),
                str(row.get("Publisher") or "").strip(),
            ))
    return items


async def _software_posix() -> list[tuple[str, str, str]]:
    if sys.platform == "darwin":
        items = []
        for directory in ("/Applications", "/System/Applications", str(Path.home() / "Applications")):
            folder = Path(directory)
            if not folder.is_dir():
                continue
            with contextlib.suppress(OSError):
                for app in sorted(folder.glob("*.app")):
                    items.append((app.stem, "", "macOS 应用"))
        return items
    dpkg = shutil.which("dpkg-query")
    if dpkg:
        out = await _run(_exe_args(dpkg, ["-W", "-f=${Package}\t${Version}\n"]), timeout=20)
        items = []
        for line in out.splitlines():
            name, _, version = line.partition("\t")
            if name.strip():
                items.append((name.strip(), version.strip(), ""))
        return items
    rpm = shutil.which("rpm")
    if rpm:
        out = await _run(_exe_args(rpm, ["-qa", "--qf", "%{NAME}\t%{VERSION}\n"]), timeout=20)
        items = []
        for line in out.splitlines():
            name, _, version = line.partition("\t")
            if name.strip():
                items.append((name.strip(), version.strip(), ""))
        return items
    return []


async def _section_software(keyword: str, limit: int, _workspace: str = "") -> str:
    lines = ["【已安装软件】"]
    try:
        items = await (_software_windows() if _is_windows() else _software_posix())
    except Exception as e:  # 单节失败不应拖垮整次查询
        lines.append(f"- 枚举失败: {type(e).__name__}: {e}")
        return "\n".join(lines)
    dedup: dict[str, tuple[str, str, str]] = {}
    for name, version, publisher in items:
        dedup.setdefault(name.casefold(), (name, version, publisher))
    rows = sorted(dedup.values(), key=lambda row: row[0].casefold())
    if keyword:
        matched = [row for row in rows if keyword.casefold() in row[0].casefold()]
        if not matched:
            lines.append(f"- 未找到名称包含「{keyword}」的软件（共 {len(rows)} 个已安装软件）。")
            return "\n".join(lines)
        rows = matched
        head = f"- 共 {len(rows)} 个匹配「{keyword}」的软件，显示前 {min(limit, len(rows))} 个："
    else:
        head = f"- 共 {len(rows)} 个已安装软件，显示前 {min(limit, len(rows))} 个（可用 keyword 过滤）："
    if not rows:
        lines.append("- 未枚举到已安装软件（可能需要管理员权限或系统无包管理器）。")
        return "\n".join(lines)
    lines.append(head)
    for name, version, publisher in rows[:limit]:
        # 不少软件把版本号写进了 DisplayName，避免出现 "App 1.2.3 1.2.3"
        detail = name if version and version.casefold() in name.casefold() else f"{name} {version}".strip()
        if publisher:
            detail += f"（{publisher}）"
        lines.append(f"- {detail}")
    if len(rows) > limit:
        lines.append(f"- …其余 {len(rows) - limit} 个未显示，可增大 limit 或使用 keyword。")
    return "\n".join(lines)


async def _section_processes(_keyword: str, limit: int, _workspace: str = "") -> str:
    lines = ["【进程（按内存排序）】"]
    if _is_windows():
        data = await _ps_json(
            f"$all=Get-Process;"
            f"$top=@($all | Sort-Object WorkingSet64 -Descending | Select-Object -First {limit} Name,Id,WorkingSet64,CPU);"
            f"[pscustomobject]@{{total=$all.Count;items=$top}} | ConvertTo-Json -Depth 3 -Compress",
            timeout=20,
        ) or {}
        items = _as_list(data.get("items"))
        total = data.get("total")
        if total:
            lines.append(f"- 共 {total} 个进程，显示内存占用最高的 {min(limit, len(items))} 个：")
        for proc in items:
            name = proc.get("Name") or "?"
            pid = proc.get("Id") or "?"
            mem = _human_bytes(proc.get("WorkingSet64"))
            cpu = proc.get("CPU")
            cpu_text = f"，CPU 累计 {float(cpu):.1f} 秒" if isinstance(cpu, (int, float)) else ""
            lines.append(f"- {name}（PID {pid}）: {mem}{cpu_text}")
        if not items:
            lines.append("- 未能读取进程列表。")
        return "\n".join(lines)
    if sys.platform == "darwin":
        out = await _run(["ps", "-Ao", "pid,comm,%cpu,%mem,rss", "-m"], timeout=10)
        rows = out.splitlines()
        lines.append(f"- 显示内存占用最高的 {min(limit, max(0, len(rows) - 1))} 个进程：")
        for line in rows[: limit + 1]:
            lines.append(f"- {line.strip()}")
        return "\n".join(lines)
    out = await _run(["ps", "-eo", "pid,comm,%cpu,%mem,rss", "--sort=-rss"], timeout=10)
    rows = out.splitlines()
    lines.append(f"- 显示内存占用最高的 {min(limit, max(0, len(rows) - 1))} 个进程：")
    for line in rows[: limit + 1]:
        lines.append(f"- {line.strip()}")
    return "\n".join(lines)


async def _section_battery(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【电池】"]
    if _is_windows():
        data = _as_list(await _ps_json(
            "Get-CimInstance Win32_Battery | Select-Object Name,EstimatedChargeRemaining,"
            "BatteryStatus,DesignCapacity,FullChargeCapacity | ConvertTo-Json -Depth 3 -Compress",
            timeout=20,
        ))
        status_map = {1: "放电中", 2: "接通电源", 3: "已充满", 4: "电量低", 5: "电量严重不足",
                      6: "充电中", 7: "充电中（高）", 8: "充电中（低）", 9: "充电中（严重不足）",
                      10: "未定义", 11: "部分充电"}
        for bat in data:
            detail = [str(bat.get("Name") or "电池")]
            if bat.get("EstimatedChargeRemaining") is not None:
                detail.append(f"电量 {bat['EstimatedChargeRemaining']}%")
            if bat.get("BatteryStatus") is not None:
                detail.append(status_map.get(bat["BatteryStatus"], f"状态码 {bat['BatteryStatus']}"))
            lines.append(f"- {'，'.join(detail)}")
        if len(lines) == 1:
            lines.append("- 未检测到电池（台式机或虚拟机）。")
        return "\n".join(lines)
    if sys.platform.startswith("linux"):
        found = False
        for battery in sorted(Path("/sys/class/power_supply").glob("BAT*")):
            try:
                capacity = (battery / "capacity").read_text().strip()
                status = (battery / "status").read_text().strip()
            except OSError:
                continue
            found = True
            lines.append(f"- {battery.name}: 电量 {capacity}%，状态 {status}")
        if not found:
            lines.append("- 未检测到电池（台式机或虚拟机）。")
        return "\n".join(lines)
    if sys.platform == "darwin":
        out = await _run(["pmset", "-g", "batt"], timeout=8)
        for line in out.splitlines():
            lines.append(f"- {line.strip()}")
        if len(lines) == 1:
            lines.append("- 未检测到电池。")
        return "\n".join(lines)
    lines.append("- 当前平台不支持电池信息读取。")
    return "\n".join(lines)


async def _section_env(_keyword: str, _limit: int, _workspace: str = "") -> str:
    lines = ["【环境变量】"]
    lines.append(f"- 变量总数: {len(os.environ)}")
    path_entries = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if path_entries:
        sample = ", ".join(path_entries[:4])
        lines.append(f"- PATH: {len(path_entries)} 项（示例: {sample}{' …' if len(path_entries) > 4 else ''}）")
    interesting = (
        "JAVA_HOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "NODE_OPTIONS",
        "COMSPEC", "SHELL", "LANG", "LC_ALL", "TEMP", "TMP", "USERPROFILE", "HOME",
        "APPDATA", "ProgramFiles", "SystemRoot",
    )
    for key in interesting:
        value = os.environ.get(key)
        if not value:
            continue
        shown = _mask_value(key, value.replace("\n", " "))
        lines.append(f"- {key}: {shown[:160]}{' …' if len(shown) > 160 else ''}")
    # PWD/OLDPWD 只是当前目录，匹配 "pwd" 规则但并非凭据，无需脱敏
    sensitive = sorted(
        key for key in os.environ
        if key.upper() not in {"PWD", "OLDPWD"} and _SENSITIVE_KEYS.search(key)
    )
    if sensitive:
        lines.append(f"- 敏感变量（已脱敏，共 {len(sensitive)} 个）:")
        for key in sensitive[:20]:
            lines.append(f"  - {key}: {_mask_value(key, os.environ.get(key, ''))}")
        if len(sensitive) > 20:
            lines.append(f"  - …其余 {len(sensitive) - 20} 个未显示")
    return "\n".join(lines)


_SECTIONS = {
    "overview": _section_overview,
    "os": _section_os,
    "cpu": _section_cpu,
    "memory": _section_memory,
    "disk": _section_disk,
    "gpu": _section_gpu,
    "network": _section_network,
    "runtimes": _section_runtimes,
    "packages": _section_packages,
    "software": _section_software,
    "processes": _section_processes,
    "battery": _section_battery,
    "env": _section_env,
}


def _parse_sections(raw: str) -> tuple[list[str], list[str]]:
    text = (raw or "").strip().lower()
    if not text:
        return ["overview"], []
    names: list[str] = []
    unknown: list[str] = []
    for token in re.split(r"[,;，；\s]+", text):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            names.extend(_SECTION_ORDER)
            continue
        target = _ALIASES.get(token, token)
        for name in target.split(","):
            if name in _SECTIONS:
                names.append(name)
            else:
                unknown.append(token)
                break
    deduped = list(dict.fromkeys(names))
    return (deduped or ["overview"]), list(dict.fromkeys(unknown))


async def _run_section(name: str, keyword: str, limit: int, workspace: str = "") -> str:
    try:
        return await _SECTIONS[name](keyword, limit, workspace)
    except Exception as e:
        return f"【{_LABELS[name]}】\n- 采集失败: {type(e).__name__}: {e}"


def _join_sections(parts: list[str]) -> str:
    blocks: list[str] = []
    total = 0
    for index, part in enumerate(parts):
        if len(part) > _MAX_SECTION_CHARS:
            part = part[:_MAX_SECTION_CHARS] + "\n- …（本节过长，已截断）"
        if total + len(part) > _MAX_TOTAL_CHARS:
            blocks.append(f"（其余 {len(parts) - index} 节因总输出上限未显示，请分次查询）")
            break
        blocks.append(part)
        total += len(part)
    return "\n\n".join(blocks)


def make_system_tools(ctx: ToolContext):

    @tool_spec(category="general", risk_level=RISK_LOW, timeout=120, idempotent=True, cacheable=False)
    @tool(parse_docstring=True)
    async def system_info(sections: str = "overview", keyword: str = "", limit: int = 30) -> str:
        """
        读取本机系统环境：操作系统、CPU/内存/磁盘/显卡、网络、运行时与依赖、已装软件、进程与环境变量（只读）

        适用：动手前需要了解当前电脑的软硬件条件，例如确认操作系统与架构、内存或磁盘余量、
        是否安装某个语言运行时/工具、显卡与驱动、本机 IP 与 DNS、已安装软件版本、确认项目里
        Vue/Vite/React/TypeScript 等依赖的实际安装版本、排查环境类报错。一次调用即可拿到结构化
        结果，不要再用 terminal 反复执行 systeminfo / ipconfig / Get-Process 等命令逐个试探。

        按节返回，默认只给概览；sections 支持逗号或空格组合：
        overview（概览）、os（系统版本/主机/管理员/开机时间）、cpu（型号/核心/主频/负载）、
        memory（物理内存/页面文件/交换分区）、disk（分区与物理磁盘）、gpu（显卡/显存/驱动）、
        network（网卡/IP/网关/DNS）、runtimes（Python/Node/Go/Rust/Java/Docker/Vite/TypeScript
        等全局命令版本）、packages（当前工作区 package.json/pyproject.toml/requirements.txt/
        Cargo.toml/go.mod 依赖清单，Node 依赖附带 node_modules 实际安装版本，另列全局 npm 包）、
        software（已安装软件，可用 keyword 过滤）、processes（按内存排序的进程）、
        battery（电池）、env（关键环境变量，敏感值脱敏）；传 all 读取全部节。

        限制与说明：全部只读，不会修改系统设置或项目文件；Windows 使用 CIM/PowerShell 采集，
        Linux/macOS 回退到 /proc、sysctl、ip、df 等命令；packages 扫描工作区及浅层子目录（跳过
        node_modules 等重型目录），Python/Cargo/Go 只报清单声明，Node 会读 node_modules 得到实际
        安装版本；全局命令只检测 PATH 内可执行文件。software、processes、packages 与 all 可能耗时
        数秒到十几秒；每节输出有上限，过长会截断并提示分次查询。环境变量中名称含
        key/secret/token/password/auth 的值会自动脱敏，不会返回明文。需要安装/升级/卸载依赖时
        用 detect_dependencies、install_dependency、uninstall_dependency；查端口占用用 list_ports，
        查项目代码用 project_outline。

        Args:
            sections: 要读取的节，逗号或空格分隔，例如 "os,cpu,packages"；默认 "overview"；all 表示全部。
            keyword: 可选，对 software 与 packages 生效：按名称过滤（不区分大小写），例如 "python"、"vite"。
            limit: 可选，software / packages 每节最多显示的条目数，默认 30，范围 1-200。

        Returns:
            分节文本（每节以【节名】开头）；未知节名会列出支持范围；某节采集失败时标注原因而不影响其他节。
        """
        ctx.check_stopped()
        names, unknown = _parse_sections(sections)
        try:
            count = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            count = 30
        kw = (keyword or "").strip()
        results = await asyncio.gather(*[_run_section(name, kw, count, ctx.workspace) for name in names])
        text = _join_sections(results)
        if unknown:
            text += f"\n\n提示：已忽略未知节 {', '.join(unknown)}；支持：{', '.join(_SECTION_ORDER)}（或 all）。"
        return text

    return (system_info,)
