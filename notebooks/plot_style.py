"""
plot_style.py — 그래프 공통 스타일 (07 노트북과 같은 톤)
"""
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

INK, SUB, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e6e5df'
NORMAL = '#c3c2b7'
TYPE_COL = {'정상': NORMAL, '미성형': '#4a3aa7', '가스+미성형': '#e87ba4', '가스': '#e34948',
            '미성형계열': '#4a3aa7', '불량': '#e34948', '오탐': INK}
TYPE_ORDER = ['미성형', '가스+미성형', '가스']
ACCENT = '#2a78d6'                      # 비교 대상(시간 맞춘 정상, 다른 Run 등)
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#ef8a88', '#b52a2a'])


def set_style():
    av = {f.name for f in font_manager.fontManager.ttflist}
    for f in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        if f in av:
            mpl.rcParams['font.family'] = f
            break
    mpl.rcParams.update({
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': SUB, 'axes.titlecolor': INK,
        'xtick.color': '#6f6e69', 'ytick.color': '#6f6e69', 'legend.frameon': False,
        'figure.dpi': 110, 'savefig.dpi': 130, 'savefig.bbox': 'tight'})


def short(c):
    """변수 이름 줄이기 (그래프 라벨용)"""
    rep = [('Max_Injection_Speed', '사출속도'), ('Max_Injection_Pressure', '사출압'),
           ('Max_Switch_Over_Pressure', '절환압'), ('Average_Back_Pressure', '배압'),
           ('Cushion_Position', '쿠션'), ('Plasticizing_Time', '계량시간'),
           ('Plasticizing_Position', '계량위치'), ('Filling_Time', '충전시간'),
           ('Injection_Time', '사출시간'), ('Cycle_Time', '사이클'), ('Clamp_Close_Time', '형체시간'),
           ('Mold_Temperature_', '금형온도'), ('Hopper_Temperature', '호퍼온도'),
           ('Barrel_Temperature_', '배럴'), ('Max_Screw_RPM', '스크류RPM'),
           ('fill_anomaly', '충전이상'), ('ev_max_abs_dev', '최대편차'), ('flow_res', '흐름저항'),
           ('_raw', ''), ('dev_', 'Δ기준선·'), ('delta_', '직전차·'),
           ('mold_', '금형·'), ('hopper_', '호퍼·')]
    for a, b in rep:
        c = c.replace(a, b)
    return c
