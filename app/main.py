from flask import Flask, request, jsonify, Response
import math
from dataclasses import dataclass
from typing import Any

@dataclass
class ThermalParameters:
    die_length: float = 0.0525
    die_width: float = 0.045
    die_thickness: float = 0.0022
    tdp: float = 150.0
    sink_length: float = 0.09
    sink_width: float = 0.116
    base_thickness: float = 0.0025
    num_fins: int = 60
    fin_thickness: float = 0.0008
    overall_height: float = 0.027
    fin_height: float = 0.0245
    k_aluminum: float = 167.0
    k_tim: float = 4.0
    tim_thickness: float = 0.0001
    k_air: float = 0.0262
    kinematic_viscosity: float = 1.568e-5
    prandtl_number: float = 0.71
    air_velocity: float = 1.0
    ambient_temp: float = 25.0
    r_jc: float = 0.2


class ThermalModel:
    LAMINAR_THRESHOLD = 2300

    def __init__(self, params: ThermalParameters | None = None, **kwargs: Any):
        if params is not None:
            self.params = params
        else:
            self.params = ThermalParameters(**kwargs)

    @property
    def die_area(self) -> float:
        return self.params.die_length * self.params.die_width

    @property
    def base_area(self) -> float:
        return self.params.sink_length * self.params.sink_width

    @property
    def fin_spacing(self) -> float:
        total_fin_width = self.params.num_fins * self.params.fin_thickness
        return (self.params.sink_width - total_fin_width) / (self.params.num_fins - 1)

    @property
    def reynolds_number(self) -> float:
        return (self.params.air_velocity * self.fin_spacing) / self.params.kinematic_viscosity

    @property
    def flow_regime(self) -> str:
        return "Laminar" if self.reynolds_number < self.LAMINAR_THRESHOLD else "Turbulent"

    @property
    def nusselt_number(self) -> float:
        re = self.reynolds_number
        pr = self.params.prandtl_number

        if re < self.LAMINAR_THRESHOLD:
            term = (re * pr * 2 * self.fin_spacing) / self.params.sink_length
            return 1.86 * math.pow(term, 1 / 3)
        else:
            return 0.023 * math.pow(re, 0.8) * math.pow(pr, 0.3)

    @property
    def convection_coefficient(self) -> float:
        return (self.nusselt_number * self.params.k_air) / (2 * self.fin_spacing)

    @property
    def single_fin_area(self) -> float:
        return (2 * self.params.fin_height + self.params.fin_thickness) * self.params.sink_length

    @property
    def total_fin_area(self) -> float:
        return self.single_fin_area * self.params.num_fins

    @property
    def base_exposed_area(self) -> float:
        return self.params.sink_length * (
            self.params.sink_width - self.params.num_fins * self.params.fin_thickness
        )

    @property
    def total_convection_area(self) -> float:
        return self.total_fin_area + self.base_exposed_area

    @property
    def r_tim(self) -> float:
        return self.params.tim_thickness / (self.params.k_tim * self.die_area)

    @property
    def r_conduction(self) -> float:
        return self.params.base_thickness / (self.params.k_aluminum * self.die_area)

    @property
    def r_convection(self) -> float:
        return 1 / (self.convection_coefficient * self.total_convection_area)

    @property
    def r_heatsink(self) -> float:
        return self.r_conduction + self.r_convection

    @property
    def total_resistance(self) -> float:
        return self.params.r_jc + self.r_tim + self.r_heatsink

    @property
    def junction_temperature(self) -> float:
        return self.params.ambient_temp + (self.params.tdp * self.total_resistance)

    def calculate(self) -> dict:
        return {
            "inputs": {
                "processor": {
                    "die_length_m": self.params.die_length,
                    "die_width_m": self.params.die_width,
                    "die_area_m2": round(self.die_area, 6),
                    "tdp_W": self.params.tdp,
                },
                "heatsink": {
                    "sink_length_m": self.params.sink_length,
                    "sink_width_m": self.params.sink_width,
                    "base_thickness_m": self.params.base_thickness,
                    "num_fins": self.params.num_fins,
                    "fin_thickness_m": self.params.fin_thickness,
                    "fin_height_m": self.params.fin_height,
                    "fin_spacing_m": round(self.fin_spacing, 6),
                    "material": "Aluminum (Al 6061-T6)",
                    "k_aluminum_W_mK": self.params.k_aluminum,
                },
                "thermal_interface": {
                    "material": "Thermal Grease",
                    "k_tim_W_mK": self.params.k_tim,
                    "thickness_m": self.params.tim_thickness,
                },
                "cooling": {
                    "medium": "Air",
                    "ambient_temp_C": self.params.ambient_temp,
                    "air_velocity_m_s": self.params.air_velocity,
                    "k_air_W_mK": self.params.k_air,
                    "kinematic_viscosity_m2_s": self.params.kinematic_viscosity,
                    "prandtl_number": self.params.prandtl_number,
                },
            },
            "results": {
                "junction_temperature_C": round(self.junction_temperature, 2),
                "total_resistance_C_W": round(self.total_resistance, 6),
            },
            "physics_debug": {
                "flow_analysis": {
                    "reynolds_number": round(self.reynolds_number, 6),
                    "flow_regime": self.flow_regime,
                    "nusselt_number": round(self.nusselt_number, 6),
                    "convection_coefficient_W_m2K": round(self.convection_coefficient, 6),
                },
                "surface_areas": {
                    "single_fin_area_m2": round(self.single_fin_area, 6),
                    "total_fin_area_m2": round(self.total_fin_area, 6),
                    "base_exposed_area_m2": round(self.base_exposed_area, 6),
                    "total_convection_area_m2": round(self.total_convection_area, 6),
                },
                "thermal_resistances": {
                    "R_jc_C_W": round(self.params.r_jc, 6),
                    "R_tim_C_W": round(self.r_tim, 6),
                    "R_cond_C_W": round(self.r_conduction, 6),
                    "R_conv_C_W": round(self.r_convection, 6),
                    "R_heatsink_C_W": round(self.r_heatsink, 6),
                    "R_total_C_W": round(self.total_resistance, 6),
                },
            },
        }

app = Flask(__name__)

print("The Thermal resistant--vivek.ky(Assignment)")


@app.route("/", methods=["GET"])
def home() -> Response:
    return jsonify({
        "service": "Heat Sink Thermal Analysis API",
        "version": "1.0.0",
        "description": "Calculate junction temperature for processor heat sink designs",
        "usage": {
            "GET /analyze": "Run analysis with default Excel reference values (TDP=150W, V=1.0 m/s)",
            "POST /analyze": "Run analysis with custom parameters via JSON body",
        },
        "example_post_body": {
            "tdp": 200,
            "air_velocity": 2.0,
            "num_fins": 80,
        },
        "reference": "Validated against Heat_Sink_Design_Ref.xlsx (Junction Temp = 80.96°C)",
    })


@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "healthy"})


@app.route("/analyze", methods=["GET"])
def analyze_default() -> Response:
    model = ThermalModel()
    return jsonify(model.calculate())


@app.route("/analyze", methods=["POST"])
def analyze_custom() -> Response:
    data = request.get_json() or {}

    allowed_params = {
        "die_length", "die_width", "die_thickness", "tdp",
        "sink_length", "sink_width", "base_thickness",
        "num_fins", "fin_thickness", "overall_height", "fin_height",
        "k_aluminum", "k_tim", "tim_thickness",
        "k_air", "kinematic_viscosity", "prandtl_number",
        "air_velocity", "ambient_temp", "r_jc",
    }

    params = {k: v for k, v in data.items() if k in allowed_params}
    model = ThermalModel(**params)
    return jsonify(model.calculate())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)