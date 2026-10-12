#!/bin/sh
# Выпуск новой версии одной командой:   ./release.sh 1.1.0
# Поднимает версию, коммитит, ставит тег и пушит. GitHub Actions собирает установщик для Windows
# и пакет для Arch, проверяет их и выкладывает в релиз. Скрипт ждёт сборку и печатает ссылку.
# Заметки к релизу берутся из CHANGELOG.md (раздел «## 1.1.0»); если его нет — из списка коммитов.
set -eu
cd "$(dirname "$0")"

V="${1:-}"
case "$V" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "Использование: ./release.sh <версия>, например ./release.sh 1.1.0" >&2; exit 1 ;;
esac
TAG="v$V"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && { echo "Тег $TAG уже есть." >&2; exit 1; }
BRANCH=$(git branch --show-current)
[ "$BRANCH" = main ] || { echo "Сейчас ветка «$BRANCH», релизы выпускаются из main." >&2; exit 1; }

sed -i "s/^VERSION = \".*\"/VERSION = \"$V\"/" maska.py
sed -i "s/^pkgver=.*/pkgver=$V/; s/^pkgrel=.*/pkgrel=1/" packaging/arch/PKGBUILD

if ! grep -q "^## $V\( \|$\)" CHANGELOG.md; then
  LAST=$(git describe --tags --abbrev=0 2>/dev/null || true)
  NOTES=$(git log --format='- %s' ${LAST:+"$LAST"..HEAD} | grep -v '^- Маска [0-9]' || true)
  [ -n "$NOTES" ] || NOTES="- Исправления и улучшения."
  NOTES="$NOTES" HEAD="## $V" awk 'NR==1 {print; print ""; print ENVIRON["HEAD"]; print ""; print ENVIRON["NOTES"]; next} {print}' \
    CHANGELOG.md > CHANGELOG.md.new && mv CHANGELOG.md.new CHANGELOG.md
  echo ":: В CHANGELOG.md добавлен раздел $V из коммитов."
fi

git add -A
git diff --cached --quiet || git commit -q -m "Маска $V"
git tag -a "$TAG" -m "Маска $V"
git push -q origin main "$TAG"
echo ":: Отправлено. Сборка Windows и Arch идёт на GitHub…"

sleep 8
RUN=$(gh run list --workflow release.yml --branch "$TAG" --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)
if [ -n "$RUN" ]; then
  gh run watch "$RUN" --exit-status >/dev/null && echo ":: Готово: $(gh release view "$TAG" --json url -q .url)" \
    || { echo "Сборка упала: gh run view $RUN --log-failed" >&2; exit 1; }
else
  echo ":: Следить за сборкой: gh run list --workflow release.yml"
fi
