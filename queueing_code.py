# queueing_code.py

import math

# If you want to copy the parse functions here again:
def parse_positive_int(value, field_name):
    try:
        ivalue = int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer.")
    if ivalue <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return ivalue

def parse_positive_float(value, field_name):
    try:
        fvalue = float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")
    if fvalue <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return fvalue


def run_queueing_model(form):
    """
    We parse the queueing distributions for arrival & service,
    then compute (λ, Ca) or (μ, Cs) + c => do M/M/c or M/G/c or G/G/c.
    """
    arr_type= form.get('q_arrival_type','poisson').lower()
    srv_type= form.get('q_service_type','exponential').lower()

    # Validate number of servers
    c= parse_positive_int(form.get('q_servers','1'), "Number of Servers (c)")

    lam, Ca= parse_queueing_arrival(form, arr_type)
    mu, Cs= parse_queueing_service(form, srv_type)

    # Identify model
    if arr_type in ['poisson','exponential']:
        arrivalMark= 'M'
    else:
        arrivalMark= 'G'
    if srv_type in ['exponential','poisson']:
        serviceMark= 'M'
    else:
        serviceMark= 'G'

    model_type= f"{arrivalMark}/{serviceMark}/{c}"
    rho= lam/(c*mu) if c>0 else 9999
    lines=[]
    lines.append(f"Queueing Model: {model_type}")
    lines.append(f"λ= {lam:.3f}, μ= {mu:.3f}, c= {c}")
    lines.append(f"Utilization(ρ)= {rho:.3f}")
    lines.append(f"(C_a= {Ca:.3f}, C_s= {Cs:.3f})")

    if arrivalMark=='M' and serviceMark=='M':
        # M/M/c
        if rho>=1:
            lines.append("System unstable(ρ>=1).")
        else:
            a= lam/mu
            sum_=0
            for k in range(c):
                sum_+=(a**k)/ math.factorial(k)
            denom= (a**c)/math.factorial(c)*(1/(1-rho))
            P0=1.0/(sum_+denom)
            Lq= (P0*(a**c)* rho)/( math.factorial(c)*((1-rho)**2))
            L= Lq+ lam/mu
            W= L/ lam
            Wq= Lq/ lam
            lines.append(f"P0= {P0:.4f}")
            lines.append(f"L= {L:.4f}, Lq= {Lq:.4f}, W= {W:.4f}, Wq= {Wq:.4f}")
    elif arrivalMark=='M' and serviceMark=='G':
        # M/G/c => approximate formula
        if rho>=1:
            lines.append("System unstable(ρ>=1).")
        else:
            Lq=(rho/(1-rho))*(((Cs**2)+1)/2)
            L= Lq+ (lam/mu)
            W= L/lam
            Wq= Lq/ lam
            lines.append("(M/G/c) used Cs^2+1.")
            lines.append(f"L= {L:.4f}, Lq= {Lq:.4f}, W= {W:.4f}, Wq= {Wq:.4f}")
    else:
        # G/G/c => approximate formula
        if rho>=1:
            lines.append("System unstable(ρ>=1).")
        else:
            Lq= (rho/(1-rho))*(((Ca**2)+(Cs**2))/2)
            L= Lq+(lam/mu)
            W= L/ lam
            Wq= Lq/ lam
            lines.append("(G/G/c) approx with Ca^2+Cs^2.")
            lines.append(f"L= {L:.4f}, Lq= {Lq:.4f}, W= {W:.4f}, Wq= {Wq:.4f}")

    return {"queueing_text":"\n".join(lines)}

def parse_queueing_arrival(form, arr_type):
    """
    Return (lambda, Ca) for arrival.
    - Poisson => param λ => lam=1/λ => Ca=1
    - Exponential => param μ => lam=1/μ => Ca=1
    - Normal => user param => (q_arrival_normal_mu, q_arrival_normal_sigma)
       => meanInter= μa => lam=1/μa => Ca= σa/ μa
    - Uniform => a,b => mean= (a+b)/2 => lam=1/mean => stdev= (b-a)/sqrt(12) => Ca= stdev/mean
    """
    if arr_type=='poisson':
        lam_input= parse_positive_float(form.get('q_arrival_poisson_lambda','1.0'),
                                        "Poisson Arrival (λ)")
        lam= 1.0/ lam_input  # Ca=1
        Ca=1.0
        return (lam,Ca)
    elif arr_type=='exponential':
        param_mu= parse_positive_float(form.get('q_arrival_exp_mu','1.0'),
                                       "Arrival Mean (μ)")
        lam=1.0/ param_mu
        Ca=1.0
        return (lam,Ca)
    elif arr_type=='normal':
        muA= parse_positive_float(form.get('q_arrival_normal_mu','5.0'),
                                  "Arrival Mean (μa)")
        sigmaA= parse_positive_float(form.get('q_arrival_normal_sigma','1.0'),
                                     "Arrival Std Dev (σa)")
        lam=1.0/muA
        Ca= sigmaA/muA
        return (lam, Ca)
    elif arr_type=='uniform':
        a_= parse_positive_float(form.get('q_arrival_uniform_a','1.0'),
                                 "Arrival Lower Bound (a)")
        b_= parse_positive_float(form.get('q_arrival_uniform_b','5.0'),
                                 "Arrival Upper Bound (b)")
        if b_ < a_:
            raise ValueError("Arrival upper bound must be >= lower bound.")
        meanI= (a_+ b_)/2.0
        lam= 1.0/meanI
        stdev= (b_-a_)/ math.sqrt(12)
        Ca= stdev/ meanI
        return (lam, Ca)
    else:
        raise ValueError("Unknown arrival dist in queueing.")

def parse_queueing_service(form, srv_type):
    """
    Return (mu, Cs).
    - Exponential => param μ => mu=1/μ => Cs=1
    - Poisson => param λ => mu=1/λ => Cs=1
    - Normal => user param => mean_s, sigma_s => mu=1/mean_s => Cs= sigma_s/mean_s
    - Uniform => a,b => mean= (a+b)/2 => mu=1/mean => stdev= (b-a)/sqrt(12) => Cs= stdev/mean
    """
    if srv_type=='exponential':
        param_mu= parse_positive_float(form.get('q_service_exp_mu','2.0'),
                                       "Service Mean (μ)")
        mu=1.0/ param_mu
        Cs=1.0
        return (mu,Cs)
    elif srv_type=='poisson':
        lam_s= parse_positive_float(form.get('q_service_poisson_lambda','2.0'),
                                    "Service Poisson (λ)")
        mu=1.0/ lam_s
        Cs=1.0
        return (mu,Cs)
    elif srv_type=='normal':
        mean_s= parse_positive_float(form.get('q_service_normal_mu','3.0'),
                                     "Service Mean (μs)")
        sigma_s= parse_positive_float(form.get('q_service_normal_sigma','1.0'),
                                      "Service Std Dev (σs)")
        mu=1.0/ mean_s
        Cs= sigma_s/ mean_s
        return (mu,Cs)
    elif srv_type=='uniform':
        a_= parse_positive_float(form.get('q_service_uniform_a','1.0'),
                                 "Service Lower Bound (a)")
        b_= parse_positive_float(form.get('q_service_uniform_b','5.0'),
                                 "Service Upper Bound (b)")
        if b_ < a_:
            raise ValueError("Service upper bound must be >= lower bound.")
        meanS= (a_+ b_)/2.0
        mu= 1.0/ meanS
        stdev= (b_-a_)/ math.sqrt(12)
        Cs= stdev/ meanS
        return (mu,Cs)
    else:
        raise ValueError("Unknown service dist in queueing.")
