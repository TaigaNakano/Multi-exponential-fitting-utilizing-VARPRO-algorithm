import numpy as np
import numpy.typing as npt

class MultiExponential:
    @staticmethod
    def sort_by_time_constants(coefficients:npt.ArrayLike, decay_coefficients:npt.ArrayLike, offset:complex|float=0.0):
        # Ensure coefficients and decay_coefficients are arrays (Idempotency)
        coefficients = np.asarray(coefficients)
        decay_coefficients = np.asarray(decay_coefficients)

        # Get the length of coefficients
        length_coeffs = len(coefficients)
        if coefficients.shape == (length_coeffs, 1) and decay_coefficients.shape == (1,length_coeffs):
            return coefficients, decay_coefficients, offset
        
        if coefficients.size != decay_coefficients.size:
            raise RuntimeError("The length of 'coefficients' and 'decay_coefficients' must be the same!!")
        
        # Sort indices by time_constants (ensures same output order on repeated calls)
        index = np.argsort(-1.0 / np.real(decay_coefficients.flatten()))[::-1]

        # Apply sorting and resizing only if necessary
        sorted_coefficients = np.resize(coefficients[index], (length_coeffs, 1))
        sorted_decay_coefficients = np.resize(decay_coefficients[index], (1, length_coeffs))

        return sorted_coefficients, sorted_decay_coefficients, offset

    def __init__(self, coefficients:npt.ArrayLike, decay_coefficients:npt.ArrayLike, offset:complex|float=0.0):
       self._exp_coeffs, self._decay_coeffs, self._offset = self.sort_by_time_constants(coefficients, decay_coefficients, offset)
    
    @staticmethod
    def eval(exp_coeffs:npt.ArrayLike, decay_coeffs:npt.ArrayLike, offset:float|complex, x:npt.ArrayLike):
        lx = np.reshape( np.asarray(x, dtype=complex), (len(x), 1))
        lexp_coeffs, ldecay_coeffs, loffset = MultiExponential.sort_by_time_constants(exp_coeffs, decay_coeffs, offset) 
        value = np.exp(lx@ldecay_coeffs)@lexp_coeffs + loffset
        return value.flatten()
       
    def __call__(self, x:npt.ArrayLike|float):
         return self.eval(self._exp_coeffs, self._decay_coeffs, self._offset, x)

    def derivative(self, x, i:int=1):
        if i < 0:
            raise ValueError("i must be non-negative!!")       
        return self.eval(self._exp_coeffs * self._decay_coeffs.T**i, self._decay_coeffs, 0.0, x)
    
    def base(self, i:int):        
        if i < 0:
            raise ValueError("i must be non-negative!!")
        elif  self.NDim < i:
            raise ValueError("i must be less than %d (0 th basis corresponds constant)"%self.NDim+1)
        else:
            if i == 0:
                return MultiExponential([1.0], [-0.0], 0.0)
            else:
                return MultiExponential([1.0], [self._decay_coeffs[0][i-1]], 0.0)
    
    @property
    def NDim(self) -> int:
        return len(self._decay_coeffs[0]) + 1
                    
    @property
    def TimeConstants(self) -> npt.NDArray:
        _LTs = -1.0/np.real( self._decay_coeffs.flatten() ) 
        return _LTs
        
    @property
    def Coefficients(self) -> npt.NDArray:        
        return self._exp_coeffs.flatten()
        
    @property
    def Offset(self) -> float:
        return self._offset
    