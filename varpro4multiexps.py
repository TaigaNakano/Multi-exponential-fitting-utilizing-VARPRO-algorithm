import  numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares 
from scipy.linalg import svd

from solver_option import SolverOption
from multi_exponentials import MultiExponential

class VarPro4MultiExponetial:
    def __init__(self, time: npt.ArrayLike, signal: npt.ArrayLike, weight: npt.ArrayLike | None = None, solver_option_path:str|None = None):
        
        if len(time) != len(signal):
            raise ValueError(f"The time array size ({time.shape[0]}) does not match the signal array size ({signal.shape[0]})")
        
        self.__time = np.asarray(time, dtype=float).reshape(-1, 1)        
        self.__signal = np.asarray(signal, dtype=float).reshape(-1, 1)
        self.__size_of_signal = self.__signal.shape[0]
        
        if weight is not None:
            weight = np.asarray(weight, dtype=float).reshape(-1)
            if weight.shape[0] != self.__size_of_signal:
                raise ValueError(f"weight size ({weight.shape[0]}) does not match signal size ({self.__size_of_signal})")
            self.__weight = weight
        else:
            self.__weight = np.ones(self.__size_of_signal, dtype=float)
        
        if solver_option_path is None:
            self.__solver_option = SolverOption()
        else:
            try:
                self.__solver_option = SolverOption.from_xml(solver_option_path)
            except Exception:
                self.__solver_option = SolverOption()

        # ----- Public data member -----
        self.verbose = 0 #verbose level in optimization w.r.t scipy.optimize.least_squares


    def __Phi(self, decay_coefficients:npt.ArrayLike) -> npt.NDArray:
        ldecay_coeffs = np.asarray(decay_coefficients, dtype=float).reshape(1,-1)
        lones = np.ones((self.__size_of_signal, 1))
        return np.append(np.exp(self.__time@ldecay_coeffs), lones, axis=1)
    

    def __intermediate_variables_for_lsq_solve(self, decay_coefficients:npt.ArrayLike):
        Phi = self.__Phi(decay_coefficients)

        # ----- 1.Compute (Phi_w := )W@Phi = U@S@Vh ------
        U, S, Vh = svd(self.__weight[:, None] * Phi, full_matrices=False)
        
        # ----- 2. Compute least square coefficients of the multi exponentials ------
        S_inv = np.divide(1, S, where=S > self.__solver_option.tols["svdtol"], out=np.zeros_like(S))
        coeffcients = (Vh.T * S_inv)@(U.T@self.__signal)

        # ----- 3. Compute wieghted residue ------
        residue =  self.__signal - Phi@coeffcients
        weighted_residue = self.__weight[:,None] * residue

        # ----- 4. Compute Jacobi matrix ------
        wt = self.__weight*self.__time.reshape(-1)  

        # --- Part A ---
        diag_c = coeffcients.reshape(-1);         
        Dkc = wt[:,None]*Phi*diag_c[None,:]                 
        A = Dkc - U@(U.T@Dkc)

        # --- Part B ---
        DkTrw = np.dot( Phi.T, wt*weighted_residue.reshape(-1) )
        B = (U*S_inv)@(Vh*DkTrw[None,:])

        jacobian = - (A + B)[:,0:-1]

        return coeffcients, residue, weighted_residue, jacobian
    
    def __lsq_solve(self, initial_time_constants):    
        lfun = lambda x: self.__intermediate_variables_for_lsq_solve(-1.0/x)[2].reshape(-1)
        ljac = lambda x: self.__intermediate_variables_for_lsq_solve(-1.0/x)[3]*(1.0/x**2)[None,:]

        result = least_squares( x0=initial_time_constants,  #fix
                                fun=lfun,                   #fix
                                jac=ljac,                   #fix
                                jac_sparsity=False,         #fix
                                verbose=self.verbose,
                                **self.__solver_option.to_dict())

        return result


    def fit(self, initial_guess:npt.ArrayLike|None = None) -> tuple[MultiExponential]:
                
        if initial_guess is None or len(initial_guess) == 0:
            raise ValueError("Initial guess must be needed!!")
      
        result = self.__lsq_solve(initial_guess)                

        if 0 < self.verbose:
            print("##### Multi-exponential estimation by the Non-Linear least square algorithm #####")
            print(result)

        decays = -1.0/result.x
        coeffs, _, _ , _ = self.__intermediate_variables_for_lsq_solve(decays)                      
        offset = coeffs[-1].item()

        return MultiExponential(coeffs[:-1], decays, offset)
    
    @property
    def SolverOption(self):
        return self.__solver_option
    

if __name__ == "__main__":
    x = np.linspace(0, 0.1,10001)

    noise_stdev = 1e-6
    noise = np.random.normal(0.0, noise_stdev, len(x))
    y = 0.1 * np.exp(-1250*x) + 0.25*np.exp(-1000*x) +  0.45*np.exp(-400*x) + 0.03
    y = y + noise

    obj = VarPro4MultiExponetial(x,y)
    obj.SolverOption.method = "trf"
    obj.verbose = 1

    exps = obj.fit([100,1000,10000])
    print(exps.Coefficients, 1.0/exps.TimeConstants, exps.Offset)



