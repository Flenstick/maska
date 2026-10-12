#!/bin/sh
# Виртуальная камера «Маска» для Discord/OBS: грузит v4l2loopback сейчас и при каждом старте системы.
set -e
[ "$(id -u)" = 0 ] || exec sudo "$0" "$@"
if ! modinfo v4l2loopback >/dev/null 2>&1; then
  echo "Модуля v4l2loopback нет в ядре. На Arch поставьте его:  sudo pacman -S v4l2loopback-dkms linux-headers" >&2
  echo "(в CachyOS он уже встроен в ядро)" >&2
  exit 1
fi
echo v4l2loopback > /etc/modules-load.d/maska-v4l2loopback.conf
echo 'options v4l2loopback devices=1 exclusive_caps=1 card_label="Маска"' > /etc/modprobe.d/maska-v4l2loopback.conf
modprobe -r v4l2loopback 2>/dev/null || true
modprobe v4l2loopback
for d in /sys/class/video4linux/video*; do
  [ "$(cat "$d/name")" = "Маска" ] && echo "Готово: виртуальная камера /dev/$(basename "$d")"
done
