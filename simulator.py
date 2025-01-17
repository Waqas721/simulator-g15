# simulator.py

import math
import random
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import factorial, exp, log, sqrt, pi
from statistics import NormalDist

def lcg_priority(count, levels):
    """Simple LCG-based priority generator (1=highest, levels=lowest)."""
    A, M, Z0, C = 55, 1994, 10112166, 9
    priorities = []
    Zi = Z0
    for _ in range(count):
        Xi = (A * Zi + C) % M
        Zi = Xi
        # Map Xi in [1..levels], 1=highest, levels=lowest
        priority = 1 + int((Xi / M) * levels)
        priorities.append(priority)
    return priorities

def pmf_poisson(k, lam):
    """Poisson PMF: P(X=k) = e^-lam * lam^k / k!."""
    if k < 0:
        return 0.0
    return (exp(-lam) * (lam**k) / math.factorial(k))

def pmf_exponential_discrete(k, lam):
    """
    Discretize Exponential(rate=lam) so that
    P(X=k) = ∫ from x=k to x=k+1 of lam * e^(-lam*x) dx.
    """
    if k < 0:
        return 0.0
    return exp(-lam * k) - exp(-lam * (k+1))

def standard_normal_cdf(x):
    """
    Approximate Φ(x) for the standard normal distribution
    without using built-in erf().
    """
    if x < 0:
        return 1.0 - standard_normal_cdf(-x)
    
    b = [ 0.319381530,
         -0.356563782,
          1.781477937,
         -1.821255978,
          1.330274429 ]
    p = 0.2316419
    
    t = 1.0 / (1.0 + p * x)
    poly = t*(b[0] + t*(b[1] + t*(b[2] + t*(b[3] + t*b[4]))))
    
    factor = math.exp(-0.5*x*x) / math.sqrt(2.0*math.pi)
    
    return 1.0 - factor*poly

def normal_cdf(x, mu, sigma):
    """
    CDF of Normal(mu, sigma) = Φ((x - mu)/sigma).
    """
    z = (x - mu) / float(sigma)
    return standard_normal_cdf(z)

def pmf_normal_discrete(k, mu, sigma):
    """
    PMF for integer k, matching Excel's:
      P(X = k) = NORM.DIST(k, mu, sigma, TRUE) - NORM.DIST(k-1, mu, sigma, TRUE).
    """
    if k < 0:
        return 0.0
    cdf_k      = normal_cdf(k, mu, sigma)
    cdf_k_minus = normal_cdf(k - 1, mu, sigma) if k > 0 else 0.0
    val = cdf_k - cdf_k_minus
    # Avoid negative due to floating-rounding:
    return max(val, 0.0)

def pmf_uniform_discrete(k, a, b):
    """
    Interpret the discrete range = round(a)..round(b).
    Then each integer in [A..B] is equally likely.
    """
    A = int(round(a))
    B = int(round(b))
    if B < A:
        return 0.0
    count_vals = B - A + 1
    if k < A or k > B:
        return 0.0
    return 1.0 / count_vals

def get_process_count(arrival_type, params):
    """
    Summation of pmf(k) from k=0..∞ until CP >= 1.0,
    with CP rounded to 16 decimal places each step.
    Then processes = k+1.
    """
    cp = 0.0
    max_k = 500  # safety cap
    count = 1

    if arrival_type == 'poisson':
        lam = params.get("arrival_rate", 1.0)
        for k in range(max_k+1):
            cp += pmf_poisson(k, lam)
            cp = round(cp, 10)  # round to 16 decimals
            if cp >= 1.0:
                count = k+1
                break

    elif arrival_type == 'exponential':
        lam = params.get("arrival_rate", 1.0)
        for k in range(max_k+1):
            cp += pmf_exponential_discrete(k, lam)
            cp = round(cp, 10)
            if cp >= 1.0:
                count = k+1
                break

    elif arrival_type == 'normal':
        mu = params.get("arrival_mu", 5.0)
        sigma = params.get("arrival_sigma", 2.0)
        for k in range(max_k + 1):
           cp += pmf_normal_discrete(k, mu, sigma)
           if cp >= 0.9999999999:
             count = k+1
             break

    elif arrival_type == 'uniform':
        a = params.get("arrival_a", 1.0)
        b = params.get("arrival_b", 5.0)
        for k in range(max_k+1):
            cp += pmf_uniform_discrete(k, a, b)
            cp = round(cp, 6)
            if cp >= 1.0:
                count = k+1
                break

    else:
        # fallback
        count = 5

    return count

def generate_interarrival_service_times(count, arrival_type, service_type, params, rate_wise):
    """
    Generate arrival, interarrival, service times (original random approach),
    once we know how many processes = 'count'.
    """
    lam = params.get("arrival_rate", 1.0)
    mu = params.get("service_rate", 1.0)

    # Rate-wise
    if rate_wise:
        arrival_time_unit = params.get("arrival_time_unit", "minute")
        service_time_unit = params.get("service_time_unit", "minute")
        if arrival_time_unit == 'hour':
            lam /= 60.0
        elif arrival_time_unit == 'second':
            lam *= 60.0
        if service_time_unit == 'hour':
            mu /= 60.0
        elif service_time_unit == 'second':
            mu *= 60.0

    arrivals = [0]
    interarrivals = [0]
    services = [1] * count

    for i in range(1, count):
        rnd_inter = random.random()
        inter_digit = int(str(rnd_inter)[2]) if len(str(rnd_inter))>2 else 1

        if arrival_type == 'poisson':
            interarrival = max(1, round(-log(rnd_inter)/lam))
        elif arrival_type == 'normal':
            # Box-Muller
            Z = sqrt(-2*log(rnd_inter))*math.cos(2*pi*random.random())
            mu_a = params.get("arrival_mu",5.0)
            sigma_a = params.get("arrival_sigma",2.0)
            interarrival = max(1, round(mu_a + sigma_a*Z))
        elif arrival_type == 'uniform':
            a_ = params.get("arrival_a",1.0)
            b_ = params.get("arrival_b",5.0)
            val = a_ + (b_ - a_)*rnd_inter
            interarrival = max(1, round(val))
        elif arrival_type == 'exponential':
            interarrival = max(1, round(-log(rnd_inter)/lam))
        else:
            raise ValueError("Invalid arrival distribution in simulator.")

        interarrival = max(interarrival, inter_digit)
        interarrivals.append(interarrival)
        arrivals.append(arrivals[-1] + interarrival)

        # Service
        rnd_service = random.random()
        serv_digit = int(str(rnd_service)[2]) if len(str(rnd_service))>2 else 1

        if service_type == 'exponential':
            service = max(1, round(-log(rnd_service)/mu))
        elif service_type == 'normal':
            Z = sqrt(-2*log(rnd_service))*math.cos(2*pi*random.random())
            mu_s = params.get("service_mu",3.0)
            sigma_s = params.get("service_sigma",1.0)
            service = max(1, round(mu_s + sigma_s*Z))
        elif service_type == 'uniform':
            sa_ = params.get("service_a",1.0)
            sb_ = params.get("service_b",5.0)
            val_s = sa_ + (sb_ - sa_)*rnd_service
            service = max(1, round(val_s))
        elif service_type == 'poisson':
            service = max(1, round(-log(rnd_service)/mu))
        else:
            raise ValueError("Invalid service distribution in simulator.")

        service = max(service, serv_digit)
        services[i] = service

    return arrivals, interarrivals, services

def simulate(arrival_type, service_type, params,
             servers, enable_priority, priority_count, rate_wise):
    """
    1) get_process_count: Summation of PMF until CP >= 1 (16 decimal precision).
    2) Generate arrival/service times for 'count' processes.
    3) Run scheduling, produce Gantt, metrics, bar charts.
    """
    count = get_process_count(arrival_type, params)

    arrivals, interarrivals, services = generate_interarrival_service_times(
        count, arrival_type, service_type, params, rate_wise
    )

    # Priority?
    if enable_priority:
        priorities = lcg_priority(count, priority_count)
    else:
        priorities = [1]*count

    start_times= [-1]*count
    end_times= [-1]*count
    turnaround_times= [0]*count
    wait_times= [0]*count
    response_times= [0]*count
    remaining_times= services[:]

    completed=0
    servers_busy= [None]*servers
    server_assignment= [-1]*count
    gantt_chart= [[] for _ in range(servers)]
    ready_queue= []
    current_time=0

    while completed< count:
        # Enqueue arrivals
        for pid in range(count):
            if arrivals[pid]== current_time:
                heappush(ready_queue,
                         (priorities[pid], arrivals[pid], pid,
                          remaining_times[pid], -1))

        # Each server
        for s in range(servers):
            pid= servers_busy[s]
            if pid is not None:
                remaining_times[pid]-= 1
                if remaining_times[pid]<=0:
                    end_times[pid]= current_time
                    turnaround_times[pid]= end_times[pid]- arrivals[pid]
                    wait_times[pid]= start_times[pid]- arrivals[pid]
                    response_times[pid]= wait_times[pid]
                    servers_busy[s]= None
                    completed+=1
                else:
                    if ready_queue:
                        top_p, top_arr, top_pid, top_rem, top_srv= ready_queue[0]
                        if top_p< priorities[pid] and (top_srv==-1 or top_srv== s):
                            heappop(ready_queue)
                            heappush(ready_queue,
                                     (priorities[pid], arrivals[pid], pid,
                                      remaining_times[pid], s))
                            server_assignment[top_pid]= s
                            servers_busy[s]= top_pid
                            if start_times[top_pid]==-1:
                                start_times[top_pid]= current_time
                            gantt_chart[s].append((top_pid, current_time, current_time+1, top_p))
                            continue
                    gantt_chart[s].append((pid, current_time, current_time+1, priorities[pid]))
            else:
                # idle
                best_index=-1
                best_priority=None
                for i, item in enumerate(ready_queue):
                    p, arr_, ppid, rrem, asrv= item
                    if asrv==-1 or asrv== s:
                        if best_priority is None or p< best_priority:
                            best_priority= p
                            best_index= i
                if best_index!=-1:
                    p, arr_, ppid, rrem, asrv= ready_queue.pop(best_index)
                    server_assignment[ppid]= s
                    servers_busy[s]= ppid
                    if start_times[ppid]==-1:
                        start_times[ppid]= current_time
                    gantt_chart[s].append((ppid, current_time, current_time+1, p))

        current_time+=1

    # Metrics
    avg_turnaround= sum(turnaround_times)/count if count>0 else 0
    avg_wait= sum(wait_times)/count if count>0 else 0
    avg_response= sum(response_times)/count if count>0 else 0
    avg_interarrival= sum(interarrivals)/ len(interarrivals) if len(interarrivals)>0 else 0
    avg_service= sum(services)/count if count>0 else 0
    total_time= max(end_times) if end_times else 1
    total_service= sum(services)
    utilization= (total_service/(servers*total_time)) if total_time>0 else 0
    server_utilization_pct= utilization*100.0

    # Priority-wise stats
    priority_stats={}
    for i in range(count):
        pri= priorities[i]
        if pri not in priority_stats:
            priority_stats[pri]= {
                'service_sum':0,'wait_sum':0,'resp_sum':0,
                'turnaround_sum':0,'count':0
            }
        priority_stats[pri]['service_sum']+= services[i]
        priority_stats[pri]['wait_sum']+= wait_times[i]
        priority_stats[pri]['resp_sum']+= response_times[i]
        priority_stats[pri]['turnaround_sum']+= turnaround_times[i]
        priority_stats[pri]['count']+=1

    priority_summary=[]
    for pri in sorted(priority_stats.keys()):
        ccc= priority_stats[pri]['count']
        if ccc>0:
            avg_wait_p= priority_stats[pri]['wait_sum']/ ccc
            avg_turnaround_p= priority_stats[pri]['turnaround_sum']/ ccc
            avg_resp_p= priority_stats[pri]['resp_sum']/ ccc
            avg_serv_p= priority_stats[pri]['service_sum']/ ccc
            priority_summary.append(
                f"Priority={pri}: Wait={avg_wait_p:.2f}, TA={avg_turnaround_p:.2f}, "
                f"Resp={avg_resp_p:.2f}, Service={avg_serv_p:.2f}"
            )
        else:
            priority_summary.append(f"Priority={pri}: No processes")

    metrics_text=[]
    metrics_text.append(f"i)   Avg Turnaround Time= {avg_turnaround:.2f}")
    metrics_text.append(f"ii)  Avg Wait Time= {avg_wait:.2f}")
    metrics_text.append(f"iii) Avg Response Time= {avg_response:.2f}")
    metrics_text.append(f"iv)  Avg Interarrival Time= {avg_interarrival:.2f}")
    metrics_text.append(f"v)   Avg Service Time= {avg_service:.2f}")
    metrics_text.append(f"vi)  Server Utilization= {server_utilization_pct:.2f}%")
    metrics_text.append("\nPriority-wise metrics:")
    for line in priority_summary:
        metrics_text.append("  "+ line)

    print("\n".join(metrics_text))

    # Print table
    if servers>1:
        if enable_priority:
            print("\nSimulation Results (1=Highest Priority) -- Multiple Servers:")
            print(f"{'Process':<8}{'Arrival':<8}{'Interarr':<9}{'Service':<8}"
                  f"{'Start':<6}{'End':<6}{'TAT':<6}{'Wait':<6}{'Resp':<6}"
                  f"{'Priority':<8}{'Server':<6}")
            for i in range(count):
                sid= server_assignment[i]
                lab= chr(ord('A')+sid) if sid!=-1 else "N/A"
                print(f"P{i+1:<7}{arrivals[i]:<8}{interarrivals[i]:<9}{services[i]:<8}"
                      f"{start_times[i]:<6}{end_times[i]:<6}"
                      f"{turnaround_times[i]:<6}{wait_times[i]:<6}{response_times[i]:<6}"
                      f"{priorities[i]:<8}{lab:<6}")
        else:
            print("\nSimulation Results (No Priority) -- Multiple Servers:")
            print(f"{'Process':<8}{'Arrival':<8}{'Interarr':<9}{'Service':<8}"
                  f"{'Start':<6}{'End':<6}{'TAT':<6}{'Wait':<6}{'Resp':<6}"
                  f"{'Server':<6}")
            for i in range(count):
                sid= server_assignment[i]
                lab= chr(ord('A')+sid) if sid!=-1 else "N/A"
                print(f"P{i+1:<7}{arrivals[i]:<8}{interarrivals[i]:<9}{services[i]:<8}"
                      f"{start_times[i]:<6}{end_times[i]:<6}"
                      f"{turnaround_times[i]:<6}{wait_times[i]:<6}{response_times[i]:<6}"
                      f"{lab:<6}")

        print("\nServer -> Processes Assignment:")
        server_procs= [[] for _ in range(servers)]
        for i in range(count):
            sid= server_assignment[i]
            if sid!=-1:
                server_procs[sid].append(f"P{i+1}")
        for s in range(servers):
            procs_str= ", ".join(server_procs[s]) if server_procs[s] else "(none)"
            print(f"  Server {chr(ord('A')+s)} served: {procs_str}")

        for s in range(servers):
            print(f"\nServer {chr(ord('A')+s)} Gantt Chart:")
            gantt_display(gantt_chart[s], enable_priority, priority_count)
    else:
        if enable_priority:
            print("\nSimulation Results (1=Highest Priority) -- Single Server:")
            print(f"{'Process':<8}{'Arrival':<8}{'Interarr':<9}{'Service':<8}"
                  f"{'Start':<6}{'End':<6}{'TAT':<6}{'Wait':<6}{'Resp':<6}{'Priority':<8}")
            for i in range(count):
                print(f"P{i+1:<7}{arrivals[i]:<8}{interarrivals[i]:<9}{services[i]:<8}"
                      f"{start_times[i]:<6}{end_times[i]:<6}"
                      f"{turnaround_times[i]:<6}{wait_times[i]:<6}{response_times[i]:<6}"
                      f"{priorities[i]:<8}")
        else:
            print("\nSimulation Results (No Priority) -- Single Server:")
            print(f"{'Process':<8}{'Arrival':<8}{'Interarr':<9}{'Service':<8}"
                  f"{'Start':<6}{'End':<6}{'TAT':<6}{'Wait':<6}{'Resp':<6}")
            for i in range(count):
                print(f"P{i+1:<7}{arrivals[i]:<8}{interarrivals[i]:<9}{services[i]:<8}"
                      f"{start_times[i]:<6}{end_times[i]:<6}"
                      f"{turnaround_times[i]:<6}{wait_times[i]:<6}{response_times[i]:<6}")
        print("\nGantt Chart (Single Server):")
        gantt_display(gantt_chart[0], enable_priority, priority_count)

    # Bar graphs
    display_bar_graphs(arrivals, interarrivals, services,
                       turnaround_times, wait_times, response_times)

def gantt_display(gantt_data, enable_priority, priority_count):
    """Draw Gantt chart. If priority scheduling, color bars based on priority."""
    if not gantt_data:
        print("  [No processes ran on this server]")
        return

    fig, ax= plt.subplots(figsize=(10,8))

    palette_5 = ["red", "orange", "yellow", "lime", "green"]
    def get_priority_color(pri):
        if priority_count<=1:
            return "blue"
        elif priority_count==2:
            return "red" if pri==1 else "green"
        elif priority_count==3:
            return {1: "red", 2: "yellow", 3: "green"}.get(pri, "blue")
        elif priority_count==4:
            return {1: "red", 2: "orange", 3: "yellow", 4:"green"}.get(pri,"blue")
        elif priority_count==5:
            idx= max(1,min(pri,5))-1
            return palette_5[idx]
        else:
            idx= (pri-1)%5
            return palette_5[idx]

    for (pid, start, end, priority) in gantt_data:
        if enable_priority:
            color= get_priority_color(priority)
        else:
            color= f"C{pid%10}"
        bar_height=4
        y_bottom= (pid+1)*8
        ax.broken_barh([(start, end-start)], (y_bottom, bar_height), facecolors=color)
        mx= start + (end-start)/2
        my= y_bottom + bar_height/2
        ax.text(mx, my, f"P{pid+1}", ha="center", va="center", color="white", fontsize=9)

    ax.set_xlabel("Time")
    ax.set_ylabel("Processes")
    all_y_positions= sorted(set((p[0]+1)*8 for p in gantt_data))
    ax.set_yticks([y+2 for y in all_y_positions])
    ax.set_yticklabels([f"P{int(y/8)}" for y in all_y_positions])
    ax.set_title("Gantt Chart")

    if enable_priority and priority_count>0:
        from matplotlib.patches import Patch
        legend_patches= []
        for pri in range(1, priority_count+1):
            legend_patches.append(Patch(color=get_priority_color(pri),
                                        label=f"Priority {pri}"))
        ax.legend(handles=legend_patches, title="Priority Colors")

    plt.show()

def display_bar_graphs(arrivals, interarrivals, services,
                       turnaround, wait, resp):
    labels= [f"P{i+1}" for i in range(len(arrivals))]
    x= range(len(arrivals))
    fig, ax= plt.subplots(3,2, figsize=(12,12))

    ax[0,0].bar(x, arrivals, color="blue")
    ax[0,0].set_title("Arrival Times")

    ax[0,1].bar(x, interarrivals, color="green")
    ax[0,1].set_title("Interarrival Times")

    ax[1,0].bar(x, services, color="red")
    ax[1,0].set_title("Service Times")

    ax[1,1].bar(x, turnaround, color="orange")
    ax[1,1].set_title("Turnaround Times")

    ax[2,0].bar(x, wait, color="purple")
    ax[2,0].set_title("Wait Times")

    ax[2,1].bar(x, resp, color="cyan")
    ax[2,1].set_title("Response Times")

    for axes in ax.flat:
        axes.set_xticks(x)
        axes.set_xticklabels(labels)
        axes.set_ylabel("Time")
        axes.set_xlabel("Processes")

    plt.tight_layout()
    plt.show()
