"""Ekstraksi fitur dari hasil parsing .lin untuk 2-stage Random Forest.

Fitur mengikuti C23 paper (Lin et al., 2023) Tabel 2, perspektif NS:
  player_hcp_[suit]   — HCP North per suit (0-10)
  partner_hcp_[suit]  — HCP South per suit (0-10)
  total_hcp           — Total HCP NS gabungan
  total_num_[suit]    — Jumlah kartu NS gabungan per suit
  player_balance      — Label distribusi North (0-3)
  partner_balance     — Label distribusi South (0-3)
  total_stop_[suit]   — Stopper NS terkuat per suit (max N,S)
  vulnerability       — Kode vulnerability (1-4)
  dealer_[seat]       — One-hot dealer position
  bid_00..bid_71      — 72-bit bidding history one-hot
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

HCP_POINTS = {"A": 4, "K": 3, "Q": 2, "J": 1}

VULN_INT = {"o": 1, "none": 1, "n": 2, "ns": 2, "e": 3, "ew": 3, "b": 4, "both": 4}

DEALER_ORDER = ["N", "E", "S", "W"]

# 72-bit bidding history (C23 Figure 1):
# 36 possible bids (P, 1C..7N) × 2 pemain (North=slot 0-35, South=slot 36-71)
_BID_LIST = ["P"] + [f"{lv}{st}" for lv in range(1, 8) for st in ["C", "D", "H", "S", "N"]]
BID_INDEX = {b: i for i, b in enumerate(_BID_LIST)}  # 36 entri


# ---------------------------------------------------------------------------
# Parsing hand
# ---------------------------------------------------------------------------

def parse_hand(hand_str: str) -> Dict[str, List[str]]:
    """Parse string hand bridge ke dict {suit: [kartu]}.

    Format input: 'SKT87HT76D873C743'
    Contoh: S='KT87', H='T76', D='873', C='743'
    """
    suits: Dict[str, List[str]] = {"S": [], "H": [], "D": [], "C": []}
    if not hand_str or not isinstance(hand_str, str):
        return suits
    parts = re.split(r"([SHDC])", hand_str.upper())
    current = None
    for part in parts:
        if part in suits:
            current = part
        elif current and part:
            suits[current] = list(part)
    return suits


# ---------------------------------------------------------------------------
# Kalkulasi fitur per pemain
# ---------------------------------------------------------------------------

def hcp_total(hand: Dict[str, List[str]]) -> int:
    """Total HCP satu hand (A=4, K=3, Q=2, J=1)."""
    return sum(HCP_POINTS.get(c, 0) for cards in hand.values() for c in cards)


def hcp_per_suit(hand: Dict[str, List[str]]) -> Dict[str, int]:
    """HCP per suit, dikap 10 sesuai C23 paper."""
    return {
        suit: min(sum(HCP_POINTS.get(c, 0) for c in hand[suit]), 10)
        for suit in ["S", "H", "D", "C"]
    }


def suit_lengths(hand: Dict[str, List[str]]) -> Dict[str, int]:
    """Jumlah kartu per suit."""
    return {suit: len(hand[suit]) for suit in ["S", "H", "D", "C"]}


def classify_balance(hand: Dict[str, List[str]]) -> int:
    """Label distribusi hand (0=seimbang, 1=cukup seimbang, 2=cukup tidak seimbang, 3=tidak seimbang).

    Balanced shapes: 4333, 4432, 5332, 5422
    """
    shape = sorted([len(hand[s]) for s in ["S", "H", "D", "C"]], reverse=True)
    t = tuple(shape)
    balanced = {(4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2), (5, 4, 2, 2)}
    if t in balanced:
        return 0
    if shape[0] == 6 and shape[1] <= 3:
        return 1
    if shape[0] >= 6 and shape[1] >= 4:
        return 2
    return 3


def classify_stopper(cards: List[str]) -> int:
    """Kekuatan stopper di satu suit.

    0=tidak ada stopper, 1=unknown, 2=stopper parsial (Q/J), 3=stopper penuh (A/Kx)
    """
    if not cards:
        return 0
    if "A" in cards:
        return 3
    if "K" in cards:
        return 3 if len(cards) >= 2 else 2
    if "Q" in cards or "J" in cards:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Bidding history encoding (C23 Figure 1)
# ---------------------------------------------------------------------------

def encode_bidding_history(bids: List[str], dealer: Optional[str]) -> List[int]:
    """Encode riwayat bidding NS sebagai vektor 72-bit one-hot.

    Mengambil bid terakhir yang bukan pass dari North (slot 0-35) dan South (slot 36-71).
    """
    vec = [0] * 72
    if not bids or dealer is None or dealer not in DEALER_ORDER:
        return vec

    idx = DEALER_ORDER.index(dealer)
    seat_order = DEALER_ORDER[idx:] + DEALER_ORDER[:idx]
    last_bid: Dict[str, str] = {"N": "P", "S": "P"}

    for i, bid in enumerate(bids):
        seat = seat_order[i % 4]
        if seat not in {"N", "S"}:
            continue
        b = bid.upper().strip("!?*+")
        if b in {"X", "XX", "PASS"}:
            b = "P"
        if b not in BID_INDEX:
            b = "P"
        if b != "P":
            last_bid[seat] = b

    vec[BID_INDEX.get(last_bid["N"], 0)] = 1
    vec[36 + BID_INDEX.get(last_bid["S"], 0)] = 1
    return vec


# ---------------------------------------------------------------------------
# Ekstraksi fitur lengkap dari satu baris parsed_boards.csv
# ---------------------------------------------------------------------------

def extract_features(row: pd.Series) -> Dict:
    """Ekstrak semua fitur C23 dari satu baris parsed_boards.csv.

    Input row harus punya kolom dari parser.py:
      north_hand_norm, south_hand_norm, east_hand_norm, west_hand_norm,
      vulnerability_code atau vulnerability,
      dealer,
      bidding_sequence
    """
    north = parse_hand(str(row.get("north_hand_norm", "")))
    south = parse_hand(str(row.get("south_hand_norm", "")))
    east  = parse_hand(str(row.get("east_hand_norm", "")))
    west  = parse_hand(str(row.get("west_hand_norm", "")))

    # Vulnerability
    vuln_raw = str(row.get("vulnerability_code", row.get("vulnerability", "o"))).lower()
    vuln_int = VULN_INT.get(vuln_raw, 1)

    # Dealer
    dealer = str(row.get("dealer", "N")).upper()

    # Bidding history
    bid_seq = str(row.get("bidding_sequence", ""))
    bids = bid_seq.split() if bid_seq else []
    bid_vec = encode_bidding_history(bids, dealer)

    # HCP per suit
    n_hcp = hcp_per_suit(north)
    s_hcp = hcp_per_suit(south)
    e_hcp = hcp_per_suit(east)
    w_hcp = hcp_per_suit(west)

    # Jumlah kartu per suit
    n_len = suit_lengths(north)
    s_len = suit_lengths(south)
    e_len = suit_lengths(east)
    w_len = suit_lengths(west)

    # Total HCP
    n_total_hcp = hcp_total(north)
    s_total_hcp = hcp_total(south)

    # NS combined
    ns_total_hcp = n_total_hcp + s_total_hcp

    # Stopper per suit
    suit_names = {"S": "spade", "H": "heart", "D": "diamond", "C": "club"}
    n_stop = {suit_names[s]: classify_stopper(north[s]) for s in ["S", "H", "D", "C"]}
    s_stop = {suit_names[s]: classify_stopper(south[s]) for s in ["S", "H", "D", "C"]}

    features: Dict = {
        # === C23 Table 2: Player (North) HCP per suit ===
        "player_hcp_spade":   n_hcp["S"],
        "player_hcp_heart":   n_hcp["H"],
        "player_hcp_diamond": n_hcp["D"],
        "player_hcp_club":    n_hcp["C"],
        # === C23 Table 2: Partner (South) HCP per suit ===
        "partner_hcp_spade":   s_hcp["S"],
        "partner_hcp_heart":   s_hcp["H"],
        "partner_hcp_diamond": s_hcp["D"],
        "partner_hcp_club":    s_hcp["C"],
        # === C23 Table 2: Total HCP NS gabungan ===
        "total_hcp": ns_total_hcp,
        # === C23 Table 2: Jumlah kartu NS gabungan per suit ===
        "total_num_spade":   n_len["S"] + s_len["S"],
        "total_num_heart":   n_len["H"] + s_len["H"],
        "total_num_diamond": n_len["D"] + s_len["D"],
        "total_num_club":    n_len["C"] + s_len["C"],
        # === C23 Table 2: Label distribusi ===
        "player_balance":  classify_balance(north),
        "partner_balance": classify_balance(south),
        # === C23 Table 2: Stopper NS (max N dan S) ===
        "total_stop_spade":   max(n_stop["spade"],   s_stop["spade"]),
        "total_stop_heart":   max(n_stop["heart"],   s_stop["heart"]),
        "total_stop_diamond": max(n_stop["diamond"], s_stop["diamond"]),
        "total_stop_club":    max(n_stop["club"],    s_stop["club"]),
        # === Vulnerability (1-4) ===
        "vulnerability": vuln_int,
        # === Dealer one-hot ===
        "dealer_N": int(dealer == "N"),
        "dealer_E": int(dealer == "E"),
        "dealer_S": int(dealer == "S"),
        "dealer_W": int(dealer == "W"),
        # === Fitur NS tambahan (keunggulan vs C23 — kita punya data kartu lengkap) ===
        "ns_best_fit": max(
            n_len["S"] + s_len["S"], n_len["H"] + s_len["H"],
            n_len["D"] + s_len["D"], n_len["C"] + s_len["C"]
        ),
        "ns_best_major_fit": max(n_len["S"] + s_len["S"], n_len["H"] + s_len["H"]),
    }

    # === 72-bit bidding history ===
    for j, bit in enumerate(bid_vec):
        features[f"bid_{j:02d}"] = bit

    return features


# ---------------------------------------------------------------------------
# Proses seluruh dataset
# ---------------------------------------------------------------------------

FEATURE_COLS = (
    [
        "player_hcp_spade", "player_hcp_heart", "player_hcp_diamond", "player_hcp_club",
        "partner_hcp_spade", "partner_hcp_heart", "partner_hcp_diamond", "partner_hcp_club",
        "total_hcp",
        "total_num_spade", "total_num_heart", "total_num_diamond", "total_num_club",
        "player_balance", "partner_balance",
        "total_stop_spade", "total_stop_heart", "total_stop_diamond", "total_stop_club",
        "vulnerability",
        "dealer_N", "dealer_E", "dealer_S", "dealer_W",
        "ns_best_fit", "ns_best_major_fit",
    ]
    + [f"bid_{j:02d}" for j in range(72)]
)


def extract_all(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Ekstrak semua fitur dari seluruh baris parsed_boards DataFrame.

    Args:
        df: DataFrame hasil parser.py (parsed_boards.csv)
        verbose: Tampilkan progress

    Returns:
        DataFrame dengan kolom-kolom fitur + kolom identitas asli
    """
    if verbose:
        print(f"Mengekstrak fitur dari {len(df)} baris...")

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        feat = extract_features(row)
        # Sertakan kolom identitas dari parser untuk traceability
        for col in ["source_file", "room", "board_number", "final_contract",
                    "tricks_claimed", "bidding_sequence"]:
            if col in df.columns:
                feat[col] = row[col]
        rows.append(feat)
        if verbose and (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} selesai")

    result = pd.DataFrame(rows)
    if verbose:
        print(f"Selesai. Shape: {result.shape}")
    return result


if __name__ == "__main__":
    parsed_csv = Path("data/parsed/parsed_boards.csv")
    if not parsed_csv.exists():
        parsed_csv = Path("data/processed/parsed_boards.csv")

    df_parsed = pd.read_csv(parsed_csv)
    print(f"Loaded {len(df_parsed)} baris dari {parsed_csv}")

    df_feat = extract_all(df_parsed)
    out = Path("data/processed/features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out, index=False)
    print(f"Fitur disimpan ke {out}")
