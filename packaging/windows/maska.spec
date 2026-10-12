# PyInstaller: сборка «Маски» для Windows (папкой — быстрее запускается, чем onefile).
# Запуск из корня репозитория:  pyinstaller packaging/windows/maska.spec
from pathlib import Path

root = Path(SPECPATH).parent.parent

a = Analysis(
    [str(root / "maska.py")],
    pathex=[str(root)],
    datas=[
        (str(root / "models"), "models"),
        (str(root / "fonts"), "fonts"),
        (str(root / "icon.svg"), "."),
        (str(root / "chevron.svg"), "."),
    ],
    hiddenimports=["pyvirtualcam", "pygrabber.dshow_graph", "PyQt6.QtSvg"],
    excludes=["v4l2out", "tkinter", "PyQt6.QtWebEngineCore", "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtMultimedia",
              "PyQt6.QtPdf", "PyQt6.Qt3DCore", "PyQt6.QtBluetooth", "PyQt6.QtSql", "PyQt6.QtTest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Maska",
    console=False,
    icon=str(root / "packaging/windows/maska.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="Maska")
