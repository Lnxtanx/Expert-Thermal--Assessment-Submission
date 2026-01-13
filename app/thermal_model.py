import math


class ThermalModel:
    def __init__(
        self,
        die_length: float = 0.0525,
        die_width: float = 0.045,
        die_thickness: float = 0.0022,
        tdp: float = 150.0,
        sink_length: float = 0.09,
        sink_width: float = 0.116,
        base_thickness: float = 0.0025,
        num_fins: int = 60,
        fin_thickness: float = 0.0008,
        overall_height: float = 0.027,
        fin_height: float = 0.0245,
        k_aluminum: float = 167.0,
        k_tim: float = 4.0,
        tim_thickness: float = 0.0001,
        k_air: float = 0.0262,
        kinematic_viscosity: float = 1.57e-5,
        prandtl_number: float = 0.71,
        air_velocity: float = 1.0,
        ambient_temp: float = 25.0,
        r_jc: float = 0.2,
    ):
        self.die_length = die_length
        self.die_width = die_width
        self.die_thickness = die_thickness
        self.tdp = tdp
        self.sink_length = sink_length
        self.sink_width = sink_width
        self.base_thickness = base_thickness
        self.num_fins = num_fins
        self.fin_thickness = fin_thickness
        self.overall_height = overall_height
        self.fin_height = fin_height
        self.k_aluminum = k_aluminum
        self.k_tim = k_tim
        self.tim_thickness = tim_thickness
        self.k_air = k_air
        self.kinematic_viscosity = kinematic_viscosity
        self.prandtl_number = prandtl_number
        self.air_velocity = air_velocity
        self.ambient_temp = ambient_temp
        self.r_jc = r_jc

    def calculate_die_area(self) -> float:
        return self.die_length * self.die_width

    def calculate_base_area(self) -> float:
        return self.sink_length * self.sink_width

    def calculate_fin_spacing(self) -> float:
        total_fin_width = self.num_fins * self.fin_thickness
        return (self.sink_width - total_fin_width) / (self.num_fins - 1)

    def calculate_r_tim(self) -> float:
        die_area = self.calculate_die_area()
        return self.tim_thickness / (self.k_tim * die_area)

    def calculate_r_conduction(self) -> float:
        die_area = self.calculate_die_area()
        return self.base_thickness / (self.k_aluminum * die_area)

    def calculate_reynolds_number(self) -> float:
        fin_spacing = self.calculate_fin_spacing()
        return (self.air_velocity * fin_spacing) / self.kinematic_viscosity

    def calculate_nusselt_number(self) -> float:
        re = self.calculate_reynolds_number()
        pr = self.prandtl_number
        fin_spacing = self.calculate_fin_spacing()

        if re < 2300:
            term = (re * pr * 2 * fin_spacing) / self.sink_length
            nu = 1.86 * math.pow(term, 1 / 3)
        else:
            nu = 0.023 * math.pow(re, 0.8) * math.pow(pr, 0.3)

        return nu

    def calculate_convection_coefficient(self) -> float:
        nu = self.calculate_nusselt_number()
        fin_spacing = self.calculate_fin_spacing()
        return (nu * self.k_air) / (2 * fin_spacing)

    def calculate_total_convection_area(self) -> float:
        single_fin_area = 2 * self.fin_height * self.sink_length
        total_fin_area = single_fin_area * self.num_fins
        base_exposed_area = self.calculate_base_area() - (
            self.num_fins * self.fin_thickness * self.sink_length
        )
        return total_fin_area + base_exposed_area

    def calculate_r_convection(self) -> float:
        h_conv = self.calculate_convection_coefficient()
        a_total = self.calculate_total_convection_area()
        return 1 / (h_conv * a_total)

    def calculate_r_heatsink(self) -> float:
        r_cond = self.calculate_r_conduction()
        r_conv = self.calculate_r_convection()
        return r_cond + r_conv

    def calculate_total_resistance(self) -> float:
        r_tim = self.calculate_r_tim()
        r_hs = self.calculate_r_heatsink()
        return self.r_jc + r_tim + r_hs

    def calculate_junction_temperature(self) -> float:
        r_total = self.calculate_total_resistance()
        return self.ambient_temp + (self.tdp * r_total)

    def get_flow_regime(self) -> str:
        re = self.calculate_reynolds_number()
        return "Laminar" if re < 2300 else "Turbulent"

    def get_full_analysis(self) -> dict:
        return {
            "input_parameters": {
                "die_length_m": self.die_length,
                "die_width_m": self.die_width,
                "die_area_m2": self.calculate_die_area(),
                "tdp_w": self.tdp,
                "sink_length_m": self.sink_length,
                "sink_width_m": self.sink_width,
                "base_thickness_m": self.base_thickness,
                "num_fins": self.num_fins,
                "fin_thickness_m": self.fin_thickness,
                "fin_height_m": self.fin_height,
                "fin_spacing_m": self.calculate_fin_spacing(),
                "ambient_temp_c": self.ambient_temp,
                "air_velocity_m_s": self.air_velocity,
            },
            "thermal_resistances": {
                "r_jc_c_per_w": self.r_jc,
                "r_tim_c_per_w": self.calculate_r_tim(),
                "r_conduction_c_per_w": self.calculate_r_conduction(),
                "r_convection_c_per_w": self.calculate_r_convection(),
                "r_heatsink_c_per_w": self.calculate_r_heatsink(),
                "r_total_c_per_w": self.calculate_total_resistance(),
            },
            "heat_transfer": {
                "reynolds_number": self.calculate_reynolds_number(),
                "flow_regime": self.get_flow_regime(),
                "nusselt_number": self.calculate_nusselt_number(),
                "convection_coefficient_w_m2k": self.calculate_convection_coefficient(),
                "total_convection_area_m2": self.calculate_total_convection_area(),
            },
            "results": {
                "junction_temperature_c": self.calculate_junction_temperature(),
            },
        }


if __name__ == "__main__":
    model = ThermalModel()
    results = model.get_full_analysis()

    print("=" * 60)
    print("THERMAL MODEL VALIDATION")
    print("=" * 60)

    print("\n--- Input Parameters ---")
    for key, value in results["input_parameters"].items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- Thermal Resistances (°C/W) ---")
    for key, value in results["thermal_resistances"].items():
        print(f"  {key}: {value:.6f}")

    print("\n--- Heat Transfer ---")
    for key, value in results["heat_transfer"].items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- Results ---")
    print(f"  Junction Temperature: {results['results']['junction_temperature_c']:.2f} °C")
    print("=" * 60)