#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${DATASET_ROOT:-}" ]]; then
  echo "[download] DATASET_ROOT no está definido. Configúralo en .env" >&2
  exit 1
fi

RAW_DIR="$DATASET_ROOT/raw"
LOG_PREFIX="[download]"

# Known datasets. Add new entries as needed.
# 1: name
# 2: type (kaggle|direct|roboflow)
# 3: source (kaggle dataset slug, direct URL, or roboflow URL)
DATASET_NAMES=(
  "PlantVillage Augmented Corn"
  "CropDG Unified Multidomain"
  "Maize Beans Tomatoes (Africa)"
  "Maize Diseases"
  "Maize In-Field Dataset"
  "Mendeley Corn Leaf Diseases"
  "Maize Nutrient Deficiency"
  "Corn Leaf (Roboflow)"
)
DATASET_TYPES=(
  "kaggle"
  "kaggle"
  "kaggle"
  "kaggle"
  "kaggle"
  "direct"
  "direct"
  "roboflow"
)
DATASET_SOURCES=(
  ""
  ""
  ""
  ""
  ""
  "https://data.mendeley.com/public-api/zip/6w6gsvghfw/download/1"
  "https://data.mendeley.com/public-api/zip/34gb2gr7p2/download/1"
  "labonis-workspace/corn-leaf-hd9iy/4"
)

print_menu() {
  echo "Available datasets:"
  echo "  0) All datasets"
  local i
  for i in "${!DATASET_NAMES[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${DATASET_NAMES[$i]}"
  done
}

ensure_raw_dir() {
  if [[ ! -d "$RAW_DIR" ]]; then
    echo "$LOG_PREFIX Creating $RAW_DIR"
    mkdir -p "$RAW_DIR"
  fi
}

check_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "$LOG_PREFIX Kaggle CLI not found. Install with: pip install kaggle"
    echo "$LOG_PREFIX Then set ~/.kaggle/kaggle.json"
    return 1
  fi
}

download_direct() {
  local url="$1"
  local out_name="$2"
  local do_unzip="$3"

  ensure_raw_dir
  mkdir -p "$RAW_DIR/$out_name"

  local zip_path="$RAW_DIR/$out_name/${out_name}.zip"
  echo "$LOG_PREFIX Downloading direct URL to $zip_path"

  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 --retry-delay 2 -o "$zip_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$zip_path" "$url"
  else
    echo "$LOG_PREFIX Neither curl nor wget found. Install one to continue."
    return 1
  fi

  if [[ "$do_unzip" == "yes" ]]; then
    echo "$LOG_PREFIX Extracting $zip_path"
    unzip -q -o "$zip_path" -d "$RAW_DIR/$out_name"
  else
    echo "$LOG_PREFIX Skipping extraction (zip kept at $zip_path)"
  fi
  echo "$LOG_PREFIX Done: $RAW_DIR/$out_name"
}

download_roboflow() {
  local slug="$1"   # workspace/project/version
  local out_name="$2"

  if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
    read -r -p "Enter your Roboflow API key (Account - Roboflow Keys): " ROBOFLOW_API_KEY
  fi
  if [[ -z "$ROBOFLOW_API_KEY" ]]; then
    echo "$LOG_PREFIX ROBOFLOW_API_KEY is required."
    return 1
  fi

  ensure_raw_dir
  local out_dir="$RAW_DIR/$out_name"
  mkdir -p "$out_dir"

  echo "$LOG_PREFIX Requesting download link for $slug"
  local api_response
  api_response="$(curl -s "https://api.roboflow.com/${slug}/yolov8?api_key=${ROBOFLOW_API_KEY}")"

  local download_url
  download_url="$(echo "$api_response" | grep -o '"link":"[^"]*"' | cut -d'"' -f4)"

  if [[ -z "$download_url" ]]; then
    echo "$LOG_PREFIX Failed to get download URL. API response:"
    echo "$api_response"
    return 1
  fi

  local zip_path="$out_dir/${out_name}.zip"
  echo "$LOG_PREFIX Downloading to $zip_path"
  curl -L "$download_url" > "$zip_path"

  echo "$LOG_PREFIX Extracting $zip_path"
  unzip "$zip_path" -d "$out_dir"
  rm "$zip_path"
  echo "$LOG_PREFIX Done: $out_dir"
}

# Placeholder Kaggle download; prompts user for dataset slug if missing.
download_kaggle() {
  local slug="$1"
  local out_name="$2"
  local do_unzip="$3"

  check_kaggle || return 1
  ensure_raw_dir

  if [[ -z "$slug" ]]; then
    read -r -p "Enter Kaggle dataset slug for '$out_name' (owner/dataset): " slug
  fi

  if [[ -z "$slug" ]]; then
    echo "$LOG_PREFIX Kaggle dataset slug is required."
    return 1
  fi

  mkdir -p "$RAW_DIR/$out_name"
  echo "$LOG_PREFIX Downloading Kaggle dataset $slug to $RAW_DIR/$out_name"
  kaggle datasets download -d "$slug" -p "$RAW_DIR/$out_name" --force
  if [[ "$do_unzip" == "yes" ]]; then
    echo "$LOG_PREFIX Extracting zip files in $RAW_DIR/$out_name"
    find "$RAW_DIR/$out_name" -maxdepth 1 -type f -name "*.zip" -print0 | while IFS= read -r -d '' z; do
      unzip -q -o "$z" -d "$RAW_DIR/$out_name"
    done
  else
    echo "$LOG_PREFIX Skipping extraction (zip files kept in $RAW_DIR/$out_name)"
  fi
  echo "$LOG_PREFIX Done: $RAW_DIR/$out_name"
}

main() {
  print_menu
  echo
  read -r -p "Select datasets by number (comma-separated), or 0/all: " selection

  if [[ -z "$selection" ]]; then
    echo "$LOG_PREFIX No selection provided."
    return 1
  fi

  local selected=()
  local unzip_choice
  local idx
  local token

  if [[ "$selection" =~ ^[Aa][Ll][Ll]$ || "$selection" == "0" ]]; then
    for idx in "${!DATASET_NAMES[@]}"; do
      selected+=("$idx")
    done
  else
    IFS=',' read -r -a tokens <<< "$selection"
    for token in "${tokens[@]}"; do
      token="$(echo "$token" | tr -d '[:space:]')"
      if [[ -z "$token" || ! "$token" =~ ^[0-9]+$ || "$token" -le 0 || "$token" -gt "${#DATASET_NAMES[@]}" ]]; then
        echo "$LOG_PREFIX Invalid selection: $token"
        return 1
      fi
      selected+=("$((token - 1))")
    done
  fi

  read -r -p "Extract zip files after download? (y/n): " unzip_choice
  unzip_choice="$(echo "$unzip_choice" | tr '[:upper:]' '[:lower:]')"
  if [[ "$unzip_choice" == "y" || "$unzip_choice" == "yes" ]]; then
    unzip_choice="yes"
  elif [[ "$unzip_choice" == "n" || "$unzip_choice" == "no" ]]; then
    unzip_choice="no"
  else
    echo "$LOG_PREFIX Invalid choice for unzip. Use y/n."
    return 1
  fi

  local s
  for s in "${selected[@]}"; do
    local name="${DATASET_NAMES[$s]}"
    local dtype="${DATASET_TYPES[$s]}"
    local source="${DATASET_SOURCES[$s]}"
    local out_dir_name

    out_dir_name="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-')"

    echo "$LOG_PREFIX Starting: $name"

    if [[ "$dtype" == "direct" ]]; then
      download_direct "$source" "$out_dir_name" "$unzip_choice"
    elif [[ "$dtype" == "roboflow" ]]; then
      download_roboflow "$source" "$out_dir_name"
    else
      download_kaggle "$source" "$out_dir_name" "$unzip_choice"
    fi
  done
}

main "$@"
