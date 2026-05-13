from .CombustionUtilities import create_CEA_object


_CEA_CACHE = {}

def get_cached_CEA(fuel_name: str, ox_name: str):
    key = (fuel_name, ox_name)
    if key not in _CEA_CACHE:
        _CEA_CACHE[key] = create_CEA_object(fuel_name, ox_name)
    return _CEA_CACHE[key]