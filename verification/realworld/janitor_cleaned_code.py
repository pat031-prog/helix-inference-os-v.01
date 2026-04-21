# Time Complexity: O(1) natively.
# Space Complexity: O(1) in-place scalar.
def calc(a: int | float, b: int | float) -> int | float:
    """
    Calculates conditional sum or pass-through.
    
    Args:
        a: Primary threshold variable.
        b: Fallback or addend variable.
    Returns:
        Sum of a+b if a > 0, otherwise returns b immediately.
    """
    return a + b if a > 0 else b