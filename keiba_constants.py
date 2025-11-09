#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競馬関連の定数定義

このファイルは競馬予想システム全体で使用する定数を定義します。
どのモジュールからでもインポートして利用できます。
"""

# 競馬場コード → 競馬場名のマッピング
TRACK_CODES = {
    '01': '札幌',
    '02': '函館', 
    '03': '福島',
    '04': '新潟',
    '05': '東京',
    '06': '中山',
    '07': '中京',
    '08': '京都',
    '09': '阪神',
    '10': '小倉'
}

# 逆引き用：競馬場名 → コード
TRACK_NAMES = {v: k for k, v in TRACK_CODES.items()}

# 競走種別コード → 名称のマッピング
KYOSO_SHUBETSU_CODES = {
    '10': '2歳',
    '11': '3歳',
    '12': '4歳以上', 
    '13': '3歳以上',
    '15': '障害'
}

# 路面種別
SURFACE_TYPES = {
    'turf': '芝',
    'dirt': 'ダート'
}

# 馬場状態コード（芝）
BABA_SHIBA_CODES = {
    '1': '良',
    '2': '稍重',
    '3': '重',
    '4': '不良'
}

# 馬場状態コード（ダート）
BABA_DIRT_CODES = {
    '1': '良',
    '2': '稍重', 
    '3': '重',
    '4': '不良'
}

# 天候コード
TENKO_CODES = {
    '1': '晴',
    '2': '曇',
    '3': '雨',
    '4': '小雨',
    '5': '雪',
    '6': '小雪'
}

# グレードコード
GRADE_CODES = {
    'A': 'G1',
    'B': 'G2', 
    'C': 'G3',
    'D': 'OP',
    'E': '1600万下',
    'F': '1000万下',
    'G': '500万下',
    'H': '未勝利'
}

# 距離カテゴリ
DISTANCE_CATEGORIES = {
    'sprint': (1000, 1400),      # スプリント
    'mile': (1401, 1800),        # マイル
    'middle': (1801, 2200),      # 中距離
    'long': (2201, 9999)         # 長距離
}

def get_track_name(keibajo_code):
    """
    競馬場コードから競馬場名を取得
    
    Args:
        keibajo_code (str): 競馬場コード
        
    Returns:
        str: 競馬場名（見つからない場合は'不明'）
    """
    return TRACK_CODES.get(keibajo_code, '不明')

def get_track_code(track_name):
    """
    競馬場名からコードを取得
    
    Args:
        track_name (str): 競馬場名
        
    Returns:
        str: 競馬場コード（見つからない場合はNone）
    """
    return TRACK_NAMES.get(track_name)

def get_distance_category(distance):
    """
    距離からカテゴリを判定
    
    Args:
        distance (int): 距離（メートル）
        
    Returns:
        str: 距離カテゴリ
    """
    for category, (min_dist, max_dist) in DISTANCE_CATEGORIES.items():
        if min_dist <= distance <= max_dist:
            return category
    return 'unknown'

def get_surface_name(surface_type):
    """
    路面種別から日本語名を取得
    
    Args:
        surface_type (str): 'turf' or 'dirt'
        
    Returns:
        str: 日本語路面名
    """
    return SURFACE_TYPES.get(surface_type, '不明')

def get_age_type_name(kyoso_shubetsu_code):
    """
    競走種別コードから年齢区分名を取得
    
    Args:
        kyoso_shubetsu_code (str): 競走種別コード
        
    Returns:
        str: 年齢区分名
    """
    return KYOSO_SHUBETSU_CODES.get(kyoso_shubetsu_code, '不明')

def format_model_description(keibajo_code, kyoso_shubetsu_code, surface_type, min_distance, max_distance):
    """
    モデルの説明文を生成
    
    Args:
        keibajo_code (str): 競馬場コード
        kyoso_shubetsu_code (str): 競走種別コード  
        surface_type (str): 路面種別
        min_distance (int): 最小距離
        max_distance (int): 最大距離
        
    Returns:
        str: モデル説明文
    """
    track_name = get_track_name(keibajo_code)
    surface_name = get_surface_name(surface_type)
    age_name = get_age_type_name(kyoso_shubetsu_code)
    
    if max_distance == 9999:
        distance_desc = f"{min_distance}m以上"
    else:
        distance_desc = f"{min_distance}-{max_distance}m"
    
    return f"{track_name}{surface_name}{distance_desc}{age_name}"

# モデル作成用の便利関数
def create_model_filename(keibajo_code, surface_type, kyoso_shubetsu_code, distance_category=None):
    """
    標準的なモデルファイル名を生成
    
    Args:
        keibajo_code (str): 競馬場コード
        surface_type (str): 路面種別
        kyoso_shubetsu_code (str): 競走種別コード
        distance_category (str, optional): 距離カテゴリ
        
    Returns:
        str: モデルファイル名
    """
    track_name_en = {
        '01': 'sapporo', '02': 'hakodate', '03': 'fukushima', '04': 'niigata',
        '05': 'tokyo', '06': 'nakayama', '07': 'chukyo', '08': 'kyoto',
        '09': 'hanshin', '10': 'kokura'
    }.get(keibajo_code, 'unknown')
    
    age_suffix = '2age' if kyoso_shubetsu_code == '10' else '3ageup'
    
    if distance_category:
        return f"{track_name_en}_{surface_type}_{age_suffix}_{distance_category}.sav"
    else:
        return f"{track_name_en}_{surface_type}_{age_suffix}.sav"