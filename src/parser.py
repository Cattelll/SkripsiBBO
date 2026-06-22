from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

BID_CONTRACT_RE = re.compile(r"^[1-7](?:C|D|H|S|N|NT)$", re.IGNORECASE)
QX_BOARD_RE = re.compile(r"^([oc])(\d+)$", re.IGNORECASE)

SUIT_SEP_RE = re.compile(r"([SHDC])", re.IGNORECASE)

DEALER_MAP = {
    "1": "S",
    "2": "W",
    "3": "N",
    "4": "E",
}

VULN_MAP = {
    "o": "none",
    "n": "ns",
    "e": "ew",
    "b": "both",
}


def parse_tokens(raw_text: str) -> List[tuple[str, str]]:
    chunks = raw_text.replace("\n", "").split("|")
    chunks = [c.strip() for c in chunks]
    if chunks and chunks[-1] == "":
        chunks.pop()

    tokens: List[tuple[str, str]] = []
    i = 0
    while i < len(chunks):
        tag = chunks[i]
        value = chunks[i + 1] if i + 1 < len(chunks) else ""
        tokens.append((tag.lower(), value))
        i += 2
    return tokens


def normalize_hand_fragment(hand_fragment: str) -> str:
    """Normalize one hand fragment into S...H...D...C... form when possible."""
    if not hand_fragment:
        return ""

    fragment = hand_fragment.strip().upper()
    if not fragment:
        return ""

    parts = SUIT_SEP_RE.split(fragment)
    # Expected shape: [prefix, suit, cards, suit, cards, ...]
    suit_map: Dict[str, str] = {"S": "", "H": "", "D": "", "C": ""}
    current_suit = None

    for part in parts:
        if not part:
            continue
        up = part.upper()
        if up in suit_map:
            current_suit = up
            continue
        if current_suit is not None:
            suit_map[current_suit] += up

    if any(suit_map.values()):
        return f"S{suit_map['S']}H{suit_map['H']}D{suit_map['D']}C{suit_map['C']}"
    return fragment


def parse_md(md_value: str) -> Dict[str, str]:
    result = {
        "dealer_code": "",
        "dealer": "",
        "south_hand_raw": "",
        "west_hand_raw": "",
        "north_hand_raw": "",
        "east_hand_raw": "",
        "south_hand_norm": "",
        "west_hand_norm": "",
        "north_hand_norm": "",
        "east_hand_norm": "",
    }

    if not md_value:
        return result

    dealer_code = md_value[0] if md_value else ""
    hands_raw = md_value[1:] if len(md_value) > 1 else ""
    parts = hands_raw.split(",")

    while len(parts) < 4:
        parts.append("")

    result["dealer_code"] = dealer_code
    result["dealer"] = DEALER_MAP.get(dealer_code, "")

    labels = ["south", "west", "north", "east"]
    for idx, label in enumerate(labels):
        raw_hand = parts[idx].strip() if idx < len(parts) else ""
        result[f"{label}_hand_raw"] = raw_hand
        result[f"{label}_hand_norm"] = normalize_hand_fragment(raw_hand)

    return result


def normalize_bid(bid: str) -> str:
    b = bid.strip().upper()
    if b == "P":
        return "PASS"
    if b == "D":
        return "X"
    if b == "R":
        return "XX"
    if b.endswith("N") and len(b) == 2 and b[0].isdigit():
        return f"{b[0]}NT"
    return b


def extract_contract(bids: List[str]) -> str:
    contracts = [b for b in bids if BID_CONTRACT_RE.match(b)]
    return contracts[-1] if contracts else "PASSOUT"


def parse_lin_file(path: Path) -> List[Dict[str, str]]:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = parse_tokens(raw_text)

    rows: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    current_bids: List[str] = []

    for tag, value in tokens:
        if tag == "qx":
            m = QX_BOARD_RE.match(value)
            if m:
                if current is not None:
                    current["bidding_sequence"] = " ".join(current_bids)
                    current["final_contract"] = extract_contract(current_bids)
                    rows.append(current)

                room_code = m.group(1).lower()
                board_no = m.group(2)
                current = {
                    "source_file": path.name,
                    "source_type": "TM" if path.name.startswith("TM-") else "BBO",
                    "room": "open" if room_code == "o" else "closed",
                    "board_number": board_no,
                    "vulnerability_code": "",
                    "vulnerability": "",
                    "dealer_code": "",
                    "dealer": "",
                    "south_hand_raw": "",
                    "west_hand_raw": "",
                    "north_hand_raw": "",
                    "east_hand_raw": "",
                    "south_hand_norm": "",
                    "west_hand_norm": "",
                    "north_hand_norm": "",
                    "east_hand_norm": "",
                    "tricks_claimed": "",
                    "bidding_sequence": "",
                    "final_contract": "",
                }
                current_bids = []
            continue

        if current is None:
            continue

        if tag == "sv":
            v = value.strip().lower()
            current["vulnerability_code"] = v
            current["vulnerability"] = VULN_MAP.get(v, "")
        elif tag == "md":
            md_info = parse_md(value.strip())
            current.update(md_info)
        elif tag == "mb":
            bid = normalize_bid(value)
            current_bids.append(bid)
        elif tag == "mc":
            current["tricks_claimed"] = value.strip()

    if current is not None:
        current["bidding_sequence"] = " ".join(current_bids)
        current["final_contract"] = extract_contract(current_bids)
        rows.append(current)

    return rows


def collect_rows(input_dir: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    files = sorted(input_dir.glob("*.lin"))
    if limit is not None:
        files = files[:limit]

    all_rows: List[Dict[str, str]] = []
    for lin_file in files:
        all_rows.extend(parse_lin_file(lin_file))
    return all_rows


def write_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "source_file",
        "source_type",
        "room",
        "board_number",
        "vulnerability_code",
        "vulnerability",
        "dealer_code",
        "dealer",
        "south_hand_raw",
        "west_hand_raw",
        "north_hand_raw",
        "east_hand_raw",
        "south_hand_norm",
        "west_hand_norm",
        "north_hand_norm",
        "east_hand_norm",
        "tricks_claimed",
        "bidding_sequence",
        "final_contract",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse BBO .lin files into board-level CSV.")
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing .lin files.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/parsed_boards.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of files to parse (for quick checks).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    rows = collect_rows(input_dir=input_dir, limit=args.limit)
    write_csv(rows=rows, output_csv=output_csv)

    print(f"Parsed {len(rows)} board records from {input_dir} -> {output_csv}")


if __name__ == "__main__":
    main()
