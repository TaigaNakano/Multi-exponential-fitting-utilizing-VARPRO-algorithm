import numpy as np
import warnings
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Tuple

class SolverOption:
    VALID_METHODS = ["trf", "dogbox", "lm"]
    VALID_LOSSES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]

    def __init__(self,
                 method: str = "trf",
                 tr_options: Optional[Dict[str, str]] = None,
                 tr_solver: Optional[str] = None,
                 bounds: Tuple[float, float] = (0.0, np.inf),
                 tols: Optional[Dict[str, float]] = None,
                 scales: Optional[Dict[str, float]] = None,
                 loss: str = "linear",
                 max_nfev: Optional[int] = None):

        self.tr_options = tr_options or {}
        self.tols = tols or {"xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12, "svdtol": 1e-12}
        self.scales = scales or {"x_scale":1.0, "f_scale": 1.0}
        self.max_nfev = max_nfev

        # 依存関係を考慮したプロパティにセット
        self.method = method
        self.tr_solver = tr_solver
        self.bounds = bounds
        self.loss = loss

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        if value not in self.VALID_METHODS:
            warnings.warn(f"Invalid method '{value}'. Using default 'trf'.")
            value = "trf"
        
        if value == "lm" and self._bounds != (-np.inf, np.inf):
            warnings.warn("Method 'lm' cannot use bounds. Resetting bounds to (-inf, inf).")
            self._bounds = (-np.inf, np.inf)

        self._method = value

    @property
    def tr_solver(self):
        return self._tr_solver

    @tr_solver.setter
    def tr_solver(self, value):
        if self._method == "dogbox" and value is not None:
            warnings.warn("Method 'dogbox' does not support 'tr_solver'. Setting 'tr_solver' to None.")
            value = None
        self._tr_solver = value

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if self._method == "lm" and value != (-np.inf, np.inf):
            warnings.warn("Method 'lm' cannot use bounds. Resetting bounds to (-inf, inf).")
            value = (-np.inf, np.inf)
        self._bounds = value

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        if value not in self.VALID_LOSSES:
            warnings.warn(f"Invalid loss '{value}'. Using default 'linear'.")
            value = "linear"
        self._loss = value

    def to_dict(self):
        options = {
            "method": self.method,
            "tr_options": self.tr_options,
            "tr_solver": self.tr_solver,
            "bounds": self.bounds,
            "xtol": self.tols.get("xtol"),
            "ftol": self.tols.get("ftol"),
            "gtol": self.tols.get("gtol"),
            "loss": self.loss,
            "f_scale": self.scales.get("f_scale"),
            "x_scale": self.scales.get("x_scale"),
            "max_nfev": self.max_nfev,
        }
        return options

    def to_xml(self, directory_path:str = "./", filename:str = "solver_option.xml" ):
        root = ET.Element("SolverOption")
        ET.SubElement(root, "Method").text = self.method
        ET.SubElement(root, "TrSolver").text = str(self.tr_solver or "")
        ET.SubElement(root, "Bounds").text = f"{self.bounds[0]},{self.bounds[1]}"
        ET.SubElement(root, "Loss").text = self.loss
        ET.SubElement(root, "MaxNfev").text = str(self.max_nfev) if self.max_nfev is not None else ""

        tols_elem = ET.SubElement(root, "Tols")
        for key, value in self.tols.items():
            ET.SubElement(tols_elem, key).text = str(value)

        scales_elem = ET.SubElement(root, "Scales")
        for key, value in self.scales.items():
            ET.SubElement(scales_elem, key).text = str(value)

        output_path = directory_path + "/" + filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ET.tostring(root, encoding="unicode"))

    @classmethod
    def from_xml(cls, path):
        """
        XML文字列からSolverOptionクラスにデシリアライズ
        """
        
        with open(path, "r", encoding="utf-8-sig") as f:
            xml_str = f.read()

        root = ET.fromstring(xml_str)
        method = root.find("Method").text
        tr_solver = root.find("TrSolver").text or None
        bounds = tuple(map(float, root.find("Bounds").text.split(',')))
        loss = root.find("Loss").text
        max_nfev = int(root.find("MaxNfev").text) if root.find("MaxNfev").text else None

        tols = {elem.tag: float(elem.text) for elem in root.find("Tols")}
        scales = {elem.tag: float(elem.text) for elem in root.find("Scales")}

        return cls(method=method, tr_solver=tr_solver, bounds=bounds, tols=tols, scales=scales, loss=loss, max_nfev=max_nfev)

    def __repr__(self):
        return f"<SolverOption(method={self.method}, loss={self.loss}, bounds={self.bounds})>"

if __name__ == "__main__":
    sol_opt = SolverOption()
    sol_opt.to_xml('./', 'solver_option.xml')