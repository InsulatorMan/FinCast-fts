import dataclasses
from dataclasses import dataclass, field

@dataclasses.dataclass(kw_only=True)
class Freq_map_dict:
    major_6_bench_map: dict = field(
        default_factory=lambda:
    {
    "ettm2": "15min",
    "ettm1": "15min",
    "etth2": "H",
    "etth1": "H",
    "electricity": "H",
    "traffic": "H",
    "weather": "10min",
    "national_illness": 'W',
    "exchange_rate": 'D',
    }
    )
    
    
    major_6_bench_val_map: dict = field(
        default_factory=lambda:
    {
    "val_elec": "H",
    "val_etth1": "H",
    "val_ettm1": "15min",
    "val_exchange": "D",
    "val_illness": "W",
    "val_traffic": "H",
    "val_weather": "10min",
    }
    )

    universal_map: dict = field(
        default_factory=lambda:
    {
    "_1d.csv": 0,
    "_1wk.csv": 1,
    "_1h.csv": 0,
    "_1m.csv": 0,
    }
    )


# def freq_map(freq: str):
#   """Returns the frequency map for the given frequency string."""
#   freq = str.upper(freq)
#   if freq.endswith("MS"):
#     return 1
#   elif freq.endswith(("H", "T", "MIN", "D", "B", "U", "S")):
#     return 0
#   elif freq.endswith(("W", "M")):
#     return 1
#   elif freq.endswith(("Y", "Q", "A")):
#     return 2
#   else:
#     raise ValueError(f"Invalid frequency: {freq}")