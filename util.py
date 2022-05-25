from functools import wraps
import brian2.numpy_ as np
from brian2.units.fundamentalunits import (
    Quantity, fail_for_dimension_mismatch, is_dimensionless, DIMENSIONLESS)

@wraps(np.concatenate)
def concatenate(arrays, /, **kwargs):
    if len(arrays) > 1:
        for array in arrays[1:]:
            fail_for_dimension_mismatch(
                arrays[0], array, 'All arguments must have the same units')
    elif len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate(arrays, **kwargs)
    if is_dimensionless(arrays[0]):
        return np.concatenate(arrays, **kwargs)
    else:
        dimensionless_arrays = [np.asarray(array) for array in arrays]
        return Quantity(
            np.concatenate(dimensionless_arrays, **kwargs),
            dim=arrays[0].dim, copy=False)


def ensure_unit(value, unit):
    if isinstance(value, dict):
        return {key: ensure_unit(val, unit) for key, val in value.items()}
    if isinstance(value, Quantity):
        # value must already be in units [unit]
        assert not isinstance(value/unit, Quantity)
    else:
        value = value * unit
    return value