from flask import Flask, request, jsonify, Response
from thermal_model import ThermalModel

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