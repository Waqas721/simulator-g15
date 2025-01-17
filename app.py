# app.py

from flask import Flask, request, jsonify, render_template
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import simulator   # This is your 'simulator.py'
import queueing_code
import sys

app = Flask(__name__)

# ---------------------------
# Helper parse/validation
# ---------------------------
def parse_positive_int(value, field_name):
    """Parses an integer and ensures it is > 0."""
    try:
        ivalue = int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer.")
    if ivalue <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return ivalue

def parse_positive_float(value, field_name):
    """Parses a float and ensures it is > 0."""
    try:
        fvalue = float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")
    if fvalue <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return fvalue

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulator')
def simapp():
    return render_template('simapp.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/run_operation', methods=['POST'])
def run_operation():
    mode = request.form.get('mode','simulate').lower()
    try:
        if mode == 'simulate':
            data = run_simulation(request.form)
            data['mode'] = 'simulate'
            return jsonify(data)
        else:
            data = run_queueing(request.form)
            data['mode'] = 'queueing'
            return jsonify(data)
    except ValueError as e:
        # Return error in JSON so front-end can show an alert
        return jsonify({"error": str(e)}), 400

def run_simulation(form):
    arrival_type = form.get('arrival_type','poisson').lower()
    service_type = form.get('service_type','exponential').lower()

    # Validate number of servers
    servers = parse_positive_int(form.get('servers','1'), "Number of Servers")

    # Priority
    enable_priority = (form.get('enable_priority','no')=='yes')
    if enable_priority:
        priority_count = parse_positive_int(
            form.get('priority_count','0'), "Priority Levels"
        )
    else:
        priority_count = 0

    # Rate-wise
    rate_wise = (form.get('rate_wise','no')=='yes')

    # Collect params
    params = {}

    # ARRIVAL
    if arrival_type == 'poisson':
        arr_lambda = parse_positive_float(form.get('arrival_lambda','1.0'),
                                          "Arrival Rate (λ)")
        params["arrival_rate"] = arr_lambda
    elif arrival_type == 'normal':
        mu_ = parse_positive_float(form.get('arrival_mu','5.0'),
                                   "Arrival Mean (μ)")
        sigma_ = parse_positive_float(form.get('arrival_sigma','2.0'),
                                      "Arrival Std Dev (σ)")
        params["arrival_mu"] = mu_
        params["arrival_sigma"] = sigma_
    elif arrival_type == 'uniform':
        a_ = parse_positive_float(form.get('arrival_a','1.0'),
                                  "Arrival Lower Bound (a)")
        b_ = parse_positive_float(form.get('arrival_b','5.0'),
                                  "Arrival Upper Bound (b)")
        if b_ < a_:
            raise ValueError("Arrival Upper Bound (b) must be >= Lower Bound (a).")
        params["arrival_a"] = a_
        params["arrival_b"] = b_
    elif arrival_type == 'exponential':
        param_mu = parse_positive_float(form.get('arrival_exp_mu','1.0'),
                                        "Arrival Mean (μ)")
        lam = 1.0 / param_mu
        params["arrival_rate"] = lam
        params["arrival_exp_mu"] = param_mu  # for reference
    else:
        raise ValueError("Invalid arrival distribution in simulation")

    # SERVICE
    if service_type == 'exponential':
        param_mu = parse_positive_float(form.get('service_rate','1.0'),
                                        "Service Mean (μ)")
        params["service_rate"] = param_mu
    elif service_type == 'normal':
        smu = parse_positive_float(form.get('service_mu','3.0'),
                                   "Service Mean (μ)")
        ssigma = parse_positive_float(form.get('service_sigma','1.0'),
                                      "Service Std Dev (σ)")
        params["service_mu"] = smu
        params["service_sigma"] = ssigma
    elif service_type == 'uniform':
        sa_ = parse_positive_float(form.get('service_a','1.0'),
                                   "Service Lower Bound (a)")
        sb_ = parse_positive_float(form.get('service_b','5.0'),
                                   "Service Upper Bound (b)")
        if sb_ < sa_:
            raise ValueError("Service Upper Bound (b) must be >= Lower Bound (a).")
        params["service_a"] = sa_
        params["service_b"] = sb_
    elif service_type == 'poisson':
        param_lam = parse_positive_float(
            form.get('service_poisson_lambda','2.0'),
            "Service Poisson (λ)"
        )
        mu_ = 1.0 / param_lam
        params["service_rate"] = mu_
    else:
        raise ValueError("Invalid service distribution in simulation")

    # Rate-wise?
    if rate_wise:
        arrival_time_unit = form.get('arrival_time_unit','minute').strip().lower()
        service_time_unit = form.get('service_time_unit','minute').strip().lower()
        params["arrival_time_unit"] = arrival_time_unit
        params["service_time_unit"] = service_time_unit
    else:
        params["arrival_time_unit"] = "minute"
        params["service_time_unit"] = "minute"

    # Now run the simulation
    sim_text, gantt_imgs, bar_img = run_complete_simulation(
        arrival_type, service_type, params,
        servers, enable_priority,
        priority_count, rate_wise
    )
    return {
        "simulation_text": sim_text,
        "gantt_images": gantt_imgs,
        "bar_charts": bar_img
    }

def run_complete_simulation(arr_type, srv_type, params,
                            servers, enable_priority,
                            priority_count, rate_wise):
    old_stdout = sys.stdout
    mystdout = io.StringIO()
    sys.stdout = mystdout

    plt.close('all')

    simulator.simulate(
        arr_type, srv_type, params,
        servers, enable_priority,
        priority_count, rate_wise
    )

    output_text = mystdout.getvalue()
    sys.stdout = old_stdout

    figs = list(map(plt.figure, plt.get_fignums()))
    gantt_images = []
    bar_charts = None
    if len(figs) > 0:
        # last figure is the bar chart
        bar_fig = figs[-1]
        bar_charts = fig_to_base64(bar_fig)

        # all prior are gantt charts
        for gf in figs[:-1]:
            gantt_images.append(fig_to_base64(gf))
    plt.close('all')
    return output_text, gantt_images, bar_charts

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def run_queueing(form):
    return queueing_code.run_queueing_model(form)

if __name__=='__main__':
    app.run(debug=True)
