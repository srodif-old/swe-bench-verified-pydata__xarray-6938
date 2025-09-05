import numpy as np
import pytest

import xarray as xr


class TestSwapDimsRegression:
    def test_swap_dims_does_not_modify_original(self):
        """Test that swap_dims does not modify the original Dataset variables.
        
        Regression test for issue where swap_dims would modify the original
        object's variable dims when creating new variables.
        """
        # Setup the test case from the issue
        nz = 11
        ds = xr.Dataset(
            data_vars={
                "y": ("z", np.random.rand(nz)),
                "lev": ("z", np.arange(nz) * 10),
            },
        )
        
        # Create ds2 as described in the issue
        ds2 = (
            ds.swap_dims(z="lev")
            .rename_dims(lev="z")
            .reset_index("lev")
            .reset_coords()
        )
        
        # Store the original dims of the lev variable
        original_lev_dims = ds2['lev'].dims
        
        # Perform the swap_dims operation that was causing the issue
        result = ds2.swap_dims(z='lev')
        
        # Check that the original ds2['lev'].dims is unchanged
        # Before the fix, this would fail because dims would be modified in place
        assert ds2['lev'].dims == original_lev_dims, (
            f"Original variable dims were modified! "
            f"Expected {original_lev_dims}, got {ds2['lev'].dims}"
        )
        
        # Also verify that the result has the correct structure
        assert 'lev' in result.dims, "Result should have 'lev' as a dimension"
        assert result['lev'].dims == ('lev',), "Result lev variable should have dims ('lev',)"