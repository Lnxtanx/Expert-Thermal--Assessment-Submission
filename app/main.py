
from flask import Flask, request, jsonify
from thermal_model import ThermalModel

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Thermal Model API",
        "endpoints": {
            "/analyze": "POST - Run thermal analysis with custom parameters",
            "/analyze/default": "GET - Run thermal analysis with default parameters",
            "/health": "GET - Health check",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/analyze/default", methods=["GET"])
def analyze_default():
    model = ThermalModel()
    return jsonify(model.get_full_analysis())


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}

    params = {
        "die_length": data.get("die_length", 0.0525),
        "die_width": data.get("die_width", 0.045),
        "die_thickness": data.get("die_thickness", 0.0022),
        "tdp": data.get("tdp", 150.0),
        "sink_length": data.get("sink_length", 0.09),
        "sink_width": data.get("sink_width", 0.116),
        "base_thickness": data.get("base_thickness", 0.0025),
        "num_fins": data.get("num_fins", 60),
        "fin_thickness": data.get("fin_thickness", 0.0008),
        "overall_height": data.get("overall_height", 0.027),
        "fin_height": data.get("fin_height", 0.0245),
        "k_aluminum": data.get("k_aluminum", 167.0),
        "k_tim": data.get("k_tim", 4.0),
        "tim_thickness": data.get("tim_thickness", 0.0001),
        "k_air": data.get("k_air", 0.0262),
        "kinematic_viscosity": data.get("kinematic_viscosity", 1.57e-5),
        "prandtl_number": data.get("prandtl_number", 0.71),
        "air_velocity": data.get("air_velocity", 1.0),
        "ambient_temp": data.get("ambient_temp", 25.0),
        "r_jc": data.get("r_jc", 0.2),
    }

    model = ThermalModel(**params)
    return jsonify(model.get_full_analysis())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)