"""Автообновление из релизов GitHub.

Windows (установленная версия): качает Maska-Setup-*.exe и запускает его тихо — установщик закрывает
программу, ставит новую версию поверх и запускает её снова.
Arch (пакет): качает .pkg.tar.zst и ставит через pkexec pacman -U.
Портативная версия и запуск из исходников: только сообщает о новой версии и открывает страницу релиза.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

REPO = "Flenstick/maska"
API = os.environ.get("MASKA_UPDATE_API", f"https://api.github.com/repos/{REPO}/releases/latest")
IS_WIN = sys.platform == "win32"


def parse(v):
    try:
        return tuple(int(x) for x in v.strip().lstrip("v").split("-")[0].split("."))
    except ValueError:
        return (0,)


def install_kind(app_dir):
    """Как поставлена программа: 'setup' (установщик Windows), 'pacman' (пакет Arch) или None — обновлять вручную."""
    if IS_WIN:
        return "setup" if getattr(sys, "frozen", False) and any(Path(sys.executable).parent.glob("unins*.exe")) else None
    if Path(app_dir).as_posix().startswith("/usr/lib/maska") and shutil.which("pkexec") and shutil.which("pacman"):
        return "pacman"
    return None


def asset_for(release, kind):
    for a in release.get("assets", []):
        name = a["name"]
        if kind == "setup" and name.startswith("Maska-Setup-") and name.endswith(".exe"):
            return a
        if kind == "pacman" and name.endswith("-x86_64.pkg.tar.zst"):
            return a
    return None


def fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Maska-updater", "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(req, timeout=timeout)


class Checker(QThread):
    """Спрашивает GitHub о последнем релизе. found(версия, url страницы, asset или None)."""
    found = pyqtSignal(str, str, object)

    def __init__(self, current, kind):
        super().__init__()
        self.current, self.kind = current, kind

    def run(self):
        try:
            with fetch(API) as r:
                rel = json.load(r)
        except Exception:  # noqa: BLE001 — нет сети, лимит API и т. п.: молча попробуем позже
            return
        tag = rel.get("tag_name", "")
        if rel.get("draft") or rel.get("prerelease") or parse(tag) <= parse(self.current):
            return
        self.found.emit(tag.lstrip("v"), rel.get("html_url", f"https://github.com/{REPO}/releases/latest"),
                        asset_for(rel, self.kind) if self.kind else None)


class Installer(QThread):
    """Качает файл обновления и запускает установку. progress(0..100), done(ошибка или '')."""
    progress = pyqtSignal(int)
    done = pyqtSignal(str)

    def __init__(self, asset, kind):
        super().__init__()
        self.asset, self.kind = asset, kind

    def run(self):
        try:
            path = Path(tempfile.gettempdir()) / self.asset["name"]
            total = self.asset.get("size") or 0
            got = 0
            with fetch(self.asset["browser_download_url"], timeout=30) as r, open(path, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        self.progress.emit(int(got * 100 / total))
            if total and got != total:
                raise OSError("файл скачался не целиком")
            if self.kind == "setup":
                # /SILENT — окно с прогрессом без вопросов; установщик сам закроет «Маску» и запустит новую
                subprocess.Popen([str(path), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                res = subprocess.run(["pkexec", "pacman", "-U", "--noconfirm", str(path)],
                                     capture_output=True, text=True)
                if res.returncode != 0:
                    raise OSError((res.stderr or res.stdout).strip().splitlines()[-1] if (res.stderr or res.stdout)
                                  else f"pacman завершился с кодом {res.returncode}")
            self.done.emit("")
        except Exception as e:  # noqa: BLE001
            self.done.emit(str(e) or e.__class__.__name__)
