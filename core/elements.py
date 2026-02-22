# core/elements.py
# Element database transcribed from sqsgenerator/include/sqsgen/core/atom.h
# Each entry: (Z, name, symbol, covalent_radius_pm, mass_amu, electronegativity)

from __future__ import annotations

_ELEMENTS_RAW = [
    (0,   "Vacancy",       "X",  0.0,    0.000,   0.00),
    (1,   "Hydrogen",      "H",  31.0,   1.008,   2.20),
    (2,   "Helium",        "He", 28.0,   4.003,   0.00),
    (3,   "Lithium",       "Li", 128.0,  6.941,   0.98),
    (4,   "Beryllium",     "Be", 96.0,   9.012,   1.57),
    (5,   "Boron",         "B",  84.0,   10.811,  2.04),
    (6,   "Carbon",        "C",  77.0,   12.011,  2.55),
    (7,   "Nitrogen",      "N",  71.0,   14.007,  3.04),
    (8,   "Oxygen",        "O",  66.0,   15.999,  3.44),
    (9,   "Fluorine",      "F",  64.0,   18.998,  3.98),
    (10,  "Neon",          "Ne", 58.0,   20.180,  0.00),
    (11,  "Sodium",        "Na", 166.0,  22.990,  0.93),
    (12,  "Magnesium",     "Mg", 141.0,  24.305,  1.31),
    (13,  "Aluminium",     "Al", 121.0,  26.982,  1.61),
    (14,  "Silicon",       "Si", 111.0,  28.086,  1.90),
    (15,  "Phosphorus",    "P",  107.0,  30.974,  2.19),
    (16,  "Sulfur",        "S",  105.0,  32.060,  2.58),
    (17,  "Chlorine",      "Cl", 102.0,  35.450,  3.16),
    (18,  "Argon",         "Ar", 106.0,  39.948,  0.00),
    (19,  "Potassium",     "K",  203.0,  39.098,  0.82),
    (20,  "Calcium",       "Ca", 176.0,  40.078,  1.00),
    (21,  "Scandium",      "Sc", 170.0,  44.956,  1.36),
    (22,  "Titanium",      "Ti", 160.0,  47.867,  1.54),
    (23,  "Vanadium",      "V",  153.0,  50.942,  1.63),
    (24,  "Chromium",      "Cr", 139.0,  51.996,  1.66),
    (25,  "Manganese",     "Mn", 139.0,  54.938,  1.55),
    (26,  "Iron",          "Fe", 132.0,  55.845,  1.83),
    (27,  "Cobalt",        "Co", 126.0,  58.933,  1.88),
    (28,  "Nickel",        "Ni", 124.0,  58.693,  1.91),
    (29,  "Copper",        "Cu", 132.0,  63.546,  1.90),
    (30,  "Zinc",          "Zn", 122.0,  65.380,  1.65),
    (31,  "Gallium",       "Ga", 122.0,  69.723,  1.81),
    (32,  "Germanium",     "Ge", 120.0,  72.630,  2.01),
    (33,  "Arsenic",       "As", 119.0,  74.922,  2.18),
    (34,  "Selenium",      "Se", 120.0,  78.971,  2.55),
    (35,  "Bromine",       "Br", 120.0,  79.904,  2.96),
    (36,  "Krypton",       "Kr", 116.0,  83.798,  3.00),
    (37,  "Rubidium",      "Rb", 220.0,  85.468,  0.82),
    (38,  "Strontium",     "Sr", 195.0,  87.620,  0.95),
    (39,  "Yttrium",       "Y",  190.0,  88.906,  1.22),
    (40,  "Zirconium",     "Zr", 175.0,  91.224,  1.33),
    (41,  "Niobium",       "Nb", 164.0,  92.906,  1.60),
    (42,  "Molybdenum",    "Mo", 154.0,  95.950,  2.16),
    (43,  "Technetium",    "Tc", 147.0,  98.000,  1.90),
    (44,  "Ruthenium",     "Ru", 146.0,  101.070, 2.20),
    (45,  "Rhodium",       "Rh", 142.0,  102.906, 2.28),
    (46,  "Palladium",     "Pd", 139.0,  106.420, 2.20),
    (47,  "Silver",        "Ag", 145.0,  107.868, 1.93),
    (48,  "Cadmium",       "Cd", 144.0,  112.411, 1.69),
    (49,  "Indium",        "In", 142.0,  114.818, 1.78),
    (50,  "Tin",           "Sn", 139.0,  118.710, 1.96),
    (51,  "Antimony",      "Sb", 139.0,  121.760, 2.05),
    (52,  "Tellurium",     "Te", 138.0,  127.600, 2.10),
    (53,  "Iodine",        "I",  139.0,  126.904, 2.66),
    (54,  "Xenon",         "Xe", 140.0,  131.293, 2.60),
    (55,  "Caesium",       "Cs", 244.0,  132.905, 0.79),
    (56,  "Barium",        "Ba", 215.0,  137.327, 0.89),
    (57,  "Lanthanum",     "La", 207.0,  138.905, 1.10),
    (58,  "Cerium",        "Ce", 204.0,  140.116, 1.12),
    (59,  "Praseodymium",  "Pr", 203.0,  140.908, 1.13),
    (60,  "Neodymium",     "Nd", 201.0,  144.242, 1.14),
    (61,  "Promethium",    "Pm", 199.0,  145.000, 1.13),
    (62,  "Samarium",      "Sm", 198.0,  150.360, 1.17),
    (63,  "Europium",      "Eu", 198.0,  151.964, 1.20),
    (64,  "Gadolinium",    "Gd", 196.0,  157.250, 1.20),
    (65,  "Terbium",       "Tb", 194.0,  158.925, 1.10),
    (66,  "Dysprosium",    "Dy", 192.0,  162.500, 1.22),
    (67,  "Holmium",       "Ho", 192.0,  164.930, 1.23),
    (68,  "Erbium",        "Er", 189.0,  167.259, 1.24),
    (69,  "Thulium",       "Tm", 190.0,  168.934, 1.25),
    (70,  "Ytterbium",     "Yb", 187.0,  173.045, 1.10),
    (71,  "Lutetium",      "Lu", 187.0,  174.967, 1.27),
    (72,  "Hafnium",       "Hf", 175.0,  178.490, 1.30),
    (73,  "Tantalum",      "Ta", 170.0,  180.948, 1.50),
    (74,  "Tungsten",      "W",  162.0,  183.840, 2.36),
    (75,  "Rhenium",       "Re", 151.0,  186.207, 1.90),
    (76,  "Osmium",        "Os", 144.0,  190.230, 2.20),
    (77,  "Iridium",       "Ir", 141.0,  192.217, 2.20),
    (78,  "Platinum",      "Pt", 136.0,  195.084, 2.28),
    (79,  "Gold",          "Au", 136.0,  196.967, 2.54),
    (80,  "Mercury",       "Hg", 132.0,  200.592, 2.00),
    (81,  "Thallium",      "Tl", 145.0,  204.383, 1.62),
    (82,  "Lead",          "Pb", 146.0,  207.200, 2.33),
    (83,  "Bismuth",       "Bi", 148.0,  208.980, 2.02),
    (84,  "Polonium",      "Po", 140.0,  209.000, 2.00),
    (85,  "Astatine",      "At", 150.0,  210.000, 2.20),
    (86,  "Radon",         "Rn", 150.0,  222.000, 0.00),
    (87,  "Francium",      "Fr", 260.0,  223.000, 0.70),
    (88,  "Radium",        "Ra", 221.0,  226.000, 0.90),
    (89,  "Actinium",      "Ac", 215.0,  227.000, 1.10),
    (90,  "Thorium",       "Th", 206.0,  232.038, 1.30),
    (91,  "Protactinium",  "Pa", 200.0,  231.036, 1.50),
    (92,  "Uranium",       "U",  196.0,  238.029, 1.38),
    (93,  "Neptunium",     "Np", 190.0,  237.000, 1.36),
    (94,  "Plutonium",     "Pu", 187.0,  244.000, 1.28),
    (95,  "Americium",     "Am", 180.0,  243.000, 1.30),
    (96,  "Curium",        "Cm", 169.0,  247.000, 1.30),
    (97,  "Berkelium",     "Bk", 0.0,    247.000, 1.30),
    (98,  "Californium",   "Cf", 0.0,    251.000, 1.30),
    (99,  "Einsteinium",   "Es", 0.0,    252.000, 1.30),
    (100, "Fermium",       "Fm", 0.0,    257.000, 1.30),
    (101, "Mendelevium",   "Md", 0.0,    258.000, 1.30),
    (102, "Nobelium",      "No", 0.0,    259.000, 1.30),
    (103, "Lawrencium",    "Lr", 0.0,    266.000, 1.30),
    (104, "Rutherfordium", "Rf", 0.0,    267.000, 0.00),
    (105, "Dubnium",       "Db", 0.0,    268.000, 0.00),
    (106, "Seaborgium",    "Sg", 0.0,    269.000, 0.00),
    (107, "Bohrium",       "Bh", 0.0,    270.000, 0.00),
    (108, "Hassium",       "Hs", 0.0,    277.000, 0.00),
    (109, "Meitnerium",    "Mt", 0.0,    278.000, 0.00),
    (110, "Darmstadtium",  "Ds", 0.0,    281.000, 0.00),
    (111, "Roentgenium",   "Rg", 0.0,    282.000, 0.00),
    (112, "Copernicium",   "Cn", 0.0,    285.000, 0.00),
    (113, "Nihonium",      "Nh", 0.0,    286.000, 0.00),
    (114, "Flerovium",     "Fl", 0.0,    289.000, 0.00),
]

# CPK-style colors (hex) — standard Jmol/CPK color scheme
_CPK_COLORS: dict[int, str] = {
    0:   "#FF69B4",  # Vacancy — pink
    1:   "#FFFFFF",  # H
    2:   "#D9FFFF",  # He
    3:   "#CC80FF",  # Li
    4:   "#C2FF00",  # Be
    5:   "#FFB5B5",  # B
    6:   "#909090",  # C
    7:   "#3050F8",  # N
    8:   "#FF0D0D",  # O
    9:   "#90E050",  # F
    10:  "#B3E3F5",  # Ne
    11:  "#AB5CF2",  # Na
    12:  "#8AFF00",  # Mg
    13:  "#BFA6A6",  # Al
    14:  "#F0C8A0",  # Si
    15:  "#FF8000",  # P
    16:  "#FFFF30",  # S
    17:  "#1FF01F",  # Cl
    18:  "#80D1E3",  # Ar
    19:  "#8F40D4",  # K
    20:  "#3DFF00",  # Ca
    21:  "#E6E6E6",  # Sc
    22:  "#BFC2C7",  # Ti
    23:  "#A6A6AB",  # V
    24:  "#8A99C7",  # Cr
    25:  "#9C7AC7",  # Mn
    26:  "#E06633",  # Fe
    27:  "#F090A0",  # Co
    28:  "#50D050",  # Ni
    29:  "#C88033",  # Cu
    30:  "#7D80B0",  # Zn
    31:  "#C28F8F",  # Ga
    32:  "#668F8F",  # Ge
    33:  "#BD80E3",  # As
    34:  "#FFA100",  # Se
    35:  "#A62929",  # Br
    36:  "#5CB8D1",  # Kr
    37:  "#702EB0",  # Rb
    38:  "#00FF00",  # Sr
    39:  "#94FFFF",  # Y
    40:  "#94E0E0",  # Zr
    41:  "#73C2C9",  # Nb
    42:  "#54B5B5",  # Mo
    43:  "#3B9E9E",  # Tc
    44:  "#248F8F",  # Ru
    45:  "#0A7D8C",  # Rh
    46:  "#006985",  # Pd
    47:  "#C0C0C0",  # Ag
    48:  "#FFD98F",  # Cd
    49:  "#A67573",  # In
    50:  "#668080",  # Sn
    51:  "#9E63B5",  # Sb
    52:  "#D47A00",  # Te
    53:  "#940094",  # I
    54:  "#429EB0",  # Xe
    55:  "#57178F",  # Cs
    56:  "#00C900",  # Ba
    57:  "#70D4FF",  # La
    58:  "#FFFFC7",  # Ce
    59:  "#D9FFC7",  # Pr
    60:  "#C7FFC7",  # Nd
    61:  "#A3FFC7",  # Pm
    62:  "#8FFFC7",  # Sm
    63:  "#61FFC7",  # Eu
    64:  "#45FFC7",  # Gd
    65:  "#30FFC7",  # Tb
    66:  "#1FFFC7",  # Dy
    67:  "#00FF9C",  # Ho
    68:  "#00E675",  # Er
    69:  "#00D452",  # Tm
    70:  "#00BF38",  # Yb
    71:  "#00AB24",  # Lu
    72:  "#4DC2FF",  # Hf
    73:  "#4DA6FF",  # Ta
    74:  "#2194D6",  # W
    75:  "#267DAB",  # Re
    76:  "#266696",  # Os
    77:  "#175487",  # Ir
    78:  "#D0D0E0",  # Pt
    79:  "#FFD123",  # Au
    80:  "#B8B8D0",  # Hg
    81:  "#A6544D",  # Tl
    82:  "#575961",  # Pb
    83:  "#9E4FB5",  # Bi
    84:  "#AB5C00",  # Po
    85:  "#754F45",  # At
    86:  "#428296",  # Rn
    87:  "#420066",  # Fr
    88:  "#007D00",  # Ra
    89:  "#70ABFA",  # Ac
    90:  "#00BAFF",  # Th
    91:  "#00A1FF",  # Pa
    92:  "#008FFF",  # U
    93:  "#0080FF",  # Np
    94:  "#006BFF",  # Pu
    95:  "#545CF2",  # Am
    96:  "#785CE3",  # Cm
    97:  "#8A4FE3",  # Bk
    98:  "#A136D4",  # Cf
    99:  "#B31FD4",  # Es
    100: "#B31FBA",  # Fm
    101: "#B30DA6",  # Md
    102: "#BD0D87",  # No
    103: "#C70066",  # Lr
    104: "#CC0059",  # Rf
    105: "#D1004F",  # Db
    106: "#D90045",  # Sg
    107: "#E00038",  # Bh
    108: "#E6002E",  # Hs
    109: "#EB0026",  # Mt
}

# Build lookup dicts at import time
_Z_MAP: dict[int, tuple] = {row[0]: row for row in _ELEMENTS_RAW}
_SYMBOL_MAP: dict[str, int] = {row[2].upper(): row[0] for row in _ELEMENTS_RAW}
# Also map lowercase and mixed case
_SYMBOL_MAP.update({row[2]: row[0] for row in _ELEMENTS_RAW})

_DEFAULT_COLOR = "#FF69B4"
_DEFAULT_RADIUS = 1.50  # Angstrom fallback


def symbol_to_z(symbol: str) -> int:
    """Convert element symbol to atomic number. Raises ValueError if unknown."""
    key = symbol.strip().capitalize()
    if key not in _SYMBOL_MAP:
        # try uppercase
        key2 = symbol.strip().upper()
        if key2 in _SYMBOL_MAP:
            return _SYMBOL_MAP[key2]
        raise ValueError(f"Unknown element symbol: '{symbol}'")
    return _SYMBOL_MAP[key]


def z_to_symbol(z: int) -> str:
    """Convert atomic number to element symbol."""
    if z not in _Z_MAP:
        raise ValueError(f"Unknown atomic number: {z}")
    return _Z_MAP[z][2]


def z_to_name(z: int) -> str:
    """Convert atomic number to element name."""
    if z not in _Z_MAP:
        raise ValueError(f"Unknown atomic number: {z}")
    return _Z_MAP[z][1]


def z_to_radius_angstrom(z: int) -> float:
    """Return covalent radius in Angstrom (converted from pm)."""
    if z not in _Z_MAP:
        return _DEFAULT_RADIUS
    r_pm = _Z_MAP[z][3]
    if r_pm <= 0:
        return _DEFAULT_RADIUS
    return r_pm / 100.0


def z_to_mass(z: int) -> float:
    """Return atomic mass in amu."""
    if z not in _Z_MAP:
        return 0.0
    return _Z_MAP[z][4]


def z_to_electronegativity(z: int) -> float:
    """Return Pauling electronegativity."""
    if z not in _Z_MAP:
        return 0.0
    return _Z_MAP[z][5]


def z_to_color(z: int) -> str:
    """Return CPK hex color string for element."""
    return _CPK_COLORS.get(z, _DEFAULT_COLOR)


def all_symbols() -> list[str]:
    """Return sorted list of all element symbols (excluding vacancy Z=0)."""
    return sorted(
        [row[2] for row in _ELEMENTS_RAW if row[0] > 0],
        key=lambda s: _SYMBOL_MAP.get(s, 999)
    )


def is_valid_symbol(symbol: str) -> bool:
    """Return True if symbol is a known element."""
    try:
        symbol_to_z(symbol)
        return True
    except ValueError:
        return False
