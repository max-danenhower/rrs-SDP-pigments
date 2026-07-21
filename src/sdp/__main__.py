from . import sdp_from_pace

pace = 'PACE_OCI.20251006T091808.L2.OC_AOP.V3_1.NRT.nc'
sdp_from_pace(pace, output_str='test_clim')
