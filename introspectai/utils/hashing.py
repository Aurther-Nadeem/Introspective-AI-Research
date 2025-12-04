import hashlib
import json

def get_hash(obj):
    """
    Returns a stable hash for a JSON-serializable object.
    """
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
