import numpy as np
import cattrs


converter = cattrs.Converter()
converter.register_unstructure_hook(float, lambda x: None if (np.isnan(x) or np.isinf(x)) else x)
converter.register_unstructure_hook(np.floating, lambda x: None if not np.isfinite(x) else float(x))
