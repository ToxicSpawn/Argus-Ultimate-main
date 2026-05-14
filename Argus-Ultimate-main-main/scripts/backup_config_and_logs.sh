#!/bin/bash
# =============================================================================
# Backup Argus config and logs (for R730 or any host)
# =============================================================================
# Usage: ./scripts/backup_config_and_logs.sh [WORK_DIR] [BACKUP_DIR]
# Example: ./scripts/backup_config_and_logs.sh /opt/argus/repo /opt/argus/backups
# Cron: 0 2 * * * /opt/argus/repo/scripts/backup_config_and_logs.sh /opt/argus/repo /opt/argus/backups
# =============================================================================

set -e

WORK_DIR="${1:-.}"
BACKUP_DIR="${2:-./backups}"
STAMP=$(date +%Y%m%d_%H%M%S)
DEST="$BACKUP_DIR/argus_backup_$STAMP"

mkdir -p "$DEST"

# Config (no secrets)
if [[ -f "$WORK_DIR/unified_config.yaml" ]]; then
   cp "$WORK_DIR/unified_config.yaml" "$DEST/"
fi
if [[ -f "$WORK_DIR/unified_config.production.yaml" ]]; then
   cp "$WORK_DIR/unified_config.production.yaml" "$DEST/"
fi

# .env redacted (copy but mask values for safety)
if [[ -f "$WORK_DIR/.env" ]]; then
   sed 's/=.*/=***REDACTED***/' "$WORK_DIR/.env" > "$DEST/.env.redacted" 2>/dev/null || true
fi

# Logs (last 7 days if dir exists)
LOG_DIR="$WORK_DIR/logs"
if [[ -d "$LOG_DIR" ]]; then
   mkdir -p "$DEST/logs"
   find "$LOG_DIR" -type f -mtime -7 -exec cp {} "$DEST/logs/" \; 2>/dev/null || true
fi
# Also common locations
for L in "/opt/argus/logs" "$WORK_DIR/../logs"; do
   if [[ -d "$L" ]]; then
      mkdir -p "$DEST/logs"
      find "$L" -type f -mtime -7 -exec cp {} "$DEST/logs/" \; 2>/dev/null || true
   fi
done

# Optional: 10gbe config
for F in "$WORK_DIR/10gbe_network_config.json" "$WORK_DIR/10gbe_network_setup.py"; do
   if [[ -f "$F" ]]; then cp "$F" "$DEST/"; fi
done

echo "[INFO] Backup written to $DEST"
# Keep only last 14 backups
if [[ -d "$BACKUP_DIR" ]]; then
   ls -dt "$BACKUP_DIR"/argus_backup_* 2>/dev/null | tail -n +15 | xargs -r rm -rf
fi
